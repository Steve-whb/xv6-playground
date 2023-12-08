// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"

/*
kernel memory layout

+------------------+ 0x88000000 (PHYSTOP)
|                  |
|    Free memory   | RW-
|                  |
+------------------+ end
|   Kernel data    | RW-
+------------------+
|   Kernel text    | R-X
+------------------+ 0x80000000 (KERNBASE)
*/

// user programs can only use the Free memory, however kernel data/text size will only be determined at runtime, hence also the case for end
// so we have to use the info that is available during compile time to decide the size of page reference count array
#define MAXNPAGES (PHYSTOP - KERNBASE)/PGSIZE
#define PA2INDEX(pa) ((uint64)pa - PGROUNDUP((uint64)end)) / PGSIZE
// Number of pages the current CPU will get from other CPU if its is empty
#define NPGTOMOVE 10
#define MIN(a, b) (a < b) ? a : b

void freerange(void *pa_start, void *pa_end);
void initlocks(void);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
  int size;
} kmems[NCPU] = {
  [0 ... NCPU-1] = { .freelist = 0, .size = 0 }
};

// only (PHYSTOP-end)/PGSIZE of entries will be used, since they corresponde to the free memroy
int page_ref_count[MAXNPAGES];
int is_initializing = 1;

void
kinit()
{
  initlocks();
  // The 1st cpu called kinit will get all physical memory initially
  freerange(end, (void*)PHYSTOP);
  memset(page_ref_count, 0, sizeof(int) * MAXNPAGES);
  is_initializing = 0;
}

void initlocks()
{
  for (int i = 0; i < NCPU; i++)
    initlock(&kmems[i].lock, "kmem");
}

void
freerange(void *pa_start, void *pa_end)
{
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}

int
is_pa_valid(uint pa)
{
  if ((pa < (uint64)end) || (pa > (uint64)PHYSTOP)) {
    printf("invalid pa: %p while access page reference count", pa);
    return -1;
  }

  return 0;
}

int
get_page_ref(uint64 pa)
{
  if (is_pa_valid(pa) == -1)
    return -1;
  
  return page_ref_count[PA2INDEX(pa)];
}

int 
inc_page_ref(uint64 pa)
{
  if (is_pa_valid(pa) == -1)
    return -1;
  
  return __sync_fetch_and_add(&page_ref_count[PA2INDEX(pa)], 1);
}

int dec_page_ref(uint64 pa)
{
  if (is_pa_valid(pa) == -1)
    return -1;
  
  return __sync_fetch_and_sub(&page_ref_count[PA2INDEX(pa)], 1);
}

// Disable interrupts to ensure a consistent execution context.
// This is necessary because a context switch (caused by an interrupt) could
// move the process to a different CPU, leading to an incorrect core number.
int 
safe_cpuid()
{
  push_off();
  int id = cpuid();
  pop_off();
  return id;
}

// Move some memory from src_cpuid to dst_cpuid. This function needs to be called with locks on 
// the two CPU's kmems acquired
int 
move_freelist(int dst_cpuid, int src_cpuid)
{
  struct run *current = kmems[src_cpuid].freelist;
  struct run *prev = 0;
  int pages_to_move = MIN(kmems[src_cpuid].size, NPGTOMOVE);
  for (int i = 0; i < pages_to_move; i++) {
      prev = current;
      current = current->next;
  }

  if (!prev || !current)
    return -1;
  
  prev->next = 0; // Detach the sublist
  kmems[dst_cpuid].freelist = kmems[src_cpuid].freelist;
  kmems[src_cpuid].freelist = current;

  kmems[dst_cpuid].size += pages_to_move;
  kmems[src_cpuid].size -= pages_to_move;
  return 0;
}

// Free the page of physical memory pointed at by pa,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(void *pa)
{
  struct run *r;
  int id;

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");

  // Only need to decrement page references count if kfree is called outside of the initialization
  if (!is_initializing) {
    if (dec_page_ref((uint64)pa) > 1)
      return;
  }

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  id = safe_cpuid();
  acquire(&kmems[id].lock);
  r->next = kmems[id].freelist;
  kmems[id].freelist = r;
  kmems[id].size++;
  release(&kmems[id].lock);
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;
  int id;

  id = safe_cpuid();
  acquire(&kmems[id].lock);

  // No free memory in the current CPU's free list, need to borrow some other CPUs
  if (kmems[id].size == 0) {
    for (int i = 0; i < NCPU; i++) {
      if (i != id && kmems[i].freelist) {
        acquire(&kmems[i].lock);
        if (move_freelist(id, i) == 0) {
          // got some free memory from i, we can start alloc memory from it
          release(&kmems[i].lock);
          break;
        } 
        release(&kmems[i].lock);
      }
    }
  }

  r = kmems[id].freelist;
  if(r) {
    kmems[id].freelist = r->next;
    kmems[id].size--;
  }
  release(&kmems[id].lock);

  if(r) {
    memset((char*)r, 5, PGSIZE); // fill with junk
    inc_page_ref((uint64)r);
  }
  return (void*)r;
}
