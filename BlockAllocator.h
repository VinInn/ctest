#ifndef BlockAllocator_H
#define BlockAllocator_H

#include<vector>
#include<map>


#include<iostream>

class BlockAllocator {
public:
  BlockAllocator( std::size_t typeSize,
		  std::size_t blockSize):
    m_typeSize(typeSize), m_blockSize(blockSize){
    init();
  }
  
  void * alloc() {
    void * ret = m_next;
    m_next+=m_typeSize;
    Block & block = *m_current;
    ++block.m_allocated;
    if(m_next==(&block.m_data.back())+1)
      init();
    return ret;
  }
  
  void dealloc(void * p) {
    pointer cp = static_cast<pointer>(p);
    // check if was last allocated
    if (cp==m_next-m_typeSize) {
      m_next=cp;
      --(*m_current).m_allocated; 
      return;
    }
    // check if in current
    Block & block = *m_current;
    if (cp>= &block.m_data.front() && cp<m_next)
      --block.m_allocated;
    else { // check the other blocks
      Blocks::iterator b = m_blocks.lower_bound(cp);
      if (cp!=(*b).first) b--;

      if ((--((*b).second->m_allocated))==0) {
	delete (*b).second;
	m_blocks.erase(b);
      }
    }
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
	       m_blocks.size()};
    return s;
  }
  
private:
  void init() {
    m_current = new Block();
    m_current->m_data.resize(m_blockSize*m_typeSize);
    m_current->m_allocated=0;
    m_next = &(m_current->m_data.front());
    m_blocks[m_next] = m_current;
  }



  struct Block {
    std::size_t m_allocated;
    std::vector<unsigned char> m_data;
  };

  typedef unsigned char * pointer; 
  typedef std::map<pointer,Block*> Blocks;

  std::size_t m_typeSize;
  std::size_t m_blockSize;
  pointer m_next;
  Block * m_current;
  Blocks m_blocks;

};

template<typename T, size_t N>
class BlockAllocated {
public:
  static void * operator new(size_t) {
    return allocator()->alloc();
  }
  
  static void operator delete(void * p) {
    allocator()->dealloc(p);
  }
  
  static BlockAllocator * allocator() {
    static BlockAllocator local(sizeof(T), N);
    return &local;
  }
  
  static BlockAllocator::Stat stat() {
    return allocator()->stat();
  }
  
private:
  
  // static BlockAllocator * s_allocator;
};


#endif // BlockAllocator_H
