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

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;

// only (PHYSTOP-end)/PGSIZE of entries will be used, since they corresponde to the free memroy
int page_ref_count[MAXNPAGES];
int is_initializing = 1;

void
kinit()
{
  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
  memset(page_ref_count, 0, sizeof(int) * MAXNPAGES);
  is_initializing = 0;
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
get_page_ref(uint64 pa)
{
  if ((pa < (uint64)end) || (pa > (uint64)PHYSTOP)) {
    printf("invalid pa while fetch page reference count: %p", pa);
    return -1;
  }
  
  return page_ref_count[PA2INDEX(pa)];
}

int 
inc_page_ref(uint64 pa)
{
  if ((pa < (uint64)end) || (pa > (uint64)PHYSTOP)) {
    printf("invalid pa while increase page reference count: %p", pa);
    return -1;
  }
  
  return __sync_fetch_and_add(&page_ref_count[PA2INDEX(pa)], 1);
}

int dec_page_ref(uint64 pa)
{
  if ((pa < (uint64)end) || (pa > (uint64)PHYSTOP)) {
    printf("invalid pa while decrease page reference count: %p", pa);
    return -1;
  }
  
  return __sync_fetch_and_sub(&page_ref_count[PA2INDEX(pa)], 1);
}

// Free the page of physical memory pointed at by pa,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(void *pa)
{
  struct run *r;

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");

  if (!is_initializing) {
    if (dec_page_ref((uint64)pa) > 1)
    return;
  }

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  acquire(&kmem.lock);
  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock);
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  if(r) {
    kmem.freelist = r->next;
  }
  release(&kmem.lock);

  if(r) {
    memset((char*)r, 5, PGSIZE); // fill with junk
    inc_page_ref((uint64)r);
  }
  return (void*)r;
}
