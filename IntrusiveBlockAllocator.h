#ifndef BlockAllocator_H
#define BlockAllocator_H

#include<vector>
#include<map>


#include<iostream>



/** to be used with a base class....
 * allows fast dealloc
 */
class IntrusiveBlockAllocator {
public:
  typedef unsigned short int Index;


  IntrusiveBlockAllocator( std::size_t typeSize,
		  std::size_t blockSize):
    m_typeSize(typeSize), m_blockSize(blockSize),nBlocks(0){
    init();
  }
  
  std::pair<void *, Block*> alloc() {
    std::pair<void *, Block*> ret(m_next, m_current);
    m_next+=m_typeSize;
    Block & block = *m_current;
    ++block.m_allocated;
    if(m_next==(&block.m_data.back())+1)
      init();
    return ret;
  }
  
  void dealloc(void * p, Block * block) {
    if (block==m_current) {
      --(*m_current).m_allocated; 
      pointer cp = static_cast<pointer>(p);
      // check if was last allocated
      if (cp==m_next-m_typeSize) m_next=cp;
      return;
    }
    //remove block if empty
    if (0==(--(*block).m_allocated) {\
	delete block;
	nBlocks++;
  }
  
public:

  struct Stat {
    size_t blockSize;
    size_t currentOccupancy;
    size_t currentAvailable;
    size_t nBlocks;
  };
  
  Stat stat() const {
    Stat s = { m_blockSize, (*m_current).m_allocated,
	       (&*(*m_current).m_data.end()-m_next)/m_typeSize,
	       nBlocks};
    return s;
  }
  
private:
  void init() {
    m_current = new Block();
    m_current->parent = this; 
    m_current->m_data.resize(m_blockSize*m_typeSize);
    m_current->m_allocated=0;
    m_next = &(m_current->m_data.front());
    nBlocks++;
  }



  struct Block {
    IntrusiveBlockAllocator * parent;
    std::size_t m_allocated;
    std::vector<unsigned char> m_data;
  };

  typedef unsigned char * pointer; 

  std::size_t m_typeSize;
  std::size_t m_blockSize;
  pointer m_next;
  Block * m_current;
  size_t nBlocks;

};


class BlockAllocated {
public:
  virtual ~BlockAllocated(){}

  static void * operator new(size_t s) {
     std::pair<void *, Block*> ret = allocator(s)->alloc();
     myblock=ret.second;
     return ret.first;
  }
  
  static void operator delete(void *) {
    myblock->parent->dealloc(myblock);
  }
  
  static IntrusiveBlockAllocator * allocator(size_t s) {
    typedef map<size_t, IntrusiveBlockAllocator *> Allocators;
    typedef Allocators::iterator AI;
    static Allocators local;
    AI p =local.find(s);
    if (p!=local.end()) return (*p).second;
    return 
      (*local.insert(std:make_pair(s, new  IntrusiveBlockAllocator(s, 1024)).second).second;
  }
  
  static BlockAllocator::Stat stat() {
    return allocator()->stat();
  }
  
private:
  IntrusiveBlockAllocator::Block * myblock;
  // static BlockAllocator * s_allocator;
};


#endif // BlockAllocator_H
