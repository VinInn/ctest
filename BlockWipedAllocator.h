#ifndef BlockWipedAllocator_H
#define BlockWipedAllocator_H

#include<vector>
#include<list>
#include<map>
#include <algorithm>

#include<boost/bind.hpp>

// #include<iostream>

/*  Allocator that never removes single allocations
 *  it "wipes" or "reset" the whole allocation when not needed anymore
 *  if not wiped it may easily run out of memory
 */
class BlockWipedAllocator {
public:
  BlockWipedAllocator( std::size_t typeSize,
		       std::size_t blockSize):
    m_typeSize(typeSize), m_blockSize(blockSize),m_alive(0){
    wipe();
  }
  

  /*  copy constructor clone the allocator and the memory it manages
   *  it needs to be reinitialized to avoid pointing to "rh"
   *
   */
  BlockWipedAllocator(BlockWipedAllocator const & rh) :
    m_typeSize(rh.m_typeSize), m_blockSize(rh.m_blockSize),m_alive(0){
    wipe();
  }

  BlockWipedAllocator& operator=(BlockWipedAllocator const & rh) {
    m_typeSize=rh.m_typeSize; m_blockSize=rh.m_blockSize; m_alive=0;
    wipe();
    return *this;
  }
    

  void * alloc() {
    m_alive++;
    void * ret = m_next;
    m_next+=m_typeSize;
    Block & block = *m_current;
    ++block.m_allocated;
    if(m_next==(&block.m_data.back())+1)
      nextBlock(true);
    return ret;
  }
  
  void dealloc(void *) {
    m_alive--;
  }

  // redime memory to the system heap
  void clear() const {
    me().m_blocks.clear();
    me().wipe();
  }

  // reset allocator status. does not redime memory
  void wipe() const {
    me().m_current=me().m_blocks.begin();
    me().nextBlock(false);
  }
  

protected:

  BlockWipedAllocator & me() const {
    return const_cast<BlockWipedAllocator&>(*this);
  }

public:

  struct Stat {
    size_t typeSize;
    size_t blockSize;
    size_t currentOccupancy;
    size_t currentAvailable;
    size_t totalAvailable;
    size_t nBlocks;
    int alive;
  };
  
  Stat stat() const {
    Stat s = { m_typeSize, m_blockSize, (*m_current).m_allocated,
	       (&*(*m_current).m_data.end()-m_next)/m_typeSize,
	       std::distance(const_iterator(m_current),m_blocks.end()),
	       m_blocks.size(), m_alive};
    return s;
  }
  
private:
  void nextBlock(bool advance) {
    if (advance) m_current++;
    if (m_current==m_blocks.end()) {
      m_blocks.push_back(Block());
      m_current=m_blocks.end(); --m_current;
    }
    m_current->m_data.resize(m_blockSize*m_typeSize);
    m_current->m_allocated=0;
    m_next = &(m_current->m_data.front());
  }



  struct Block {
    std::size_t m_allocated;
    std::vector<unsigned char> m_data;
  };

  typedef unsigned char * pointer; 
  typedef std::list<Block> Blocks;
  typedef Blocks::iterator iterator;
  typedef Blocks::const_iterator const_iterator;


  std::size_t m_typeSize;
  std::size_t m_blockSize;
  pointer m_next;
  iterator m_current;
  Blocks m_blocks;
  int m_alive;

};


class BlockWipedPool {
public:
  typedef BlockWipedAllocator Allocator;
  typedef std::map<std::size_t, Allocator> Pool; 

  BlockWipedPool(std::size_t blockSize) : m_blockSize(blockSize){}


  Allocator & allocator( std::size_t typeSize) {
    Pool::iterator p=m_pool.find(typeSize);
    if (p!=m_pool.end()) return (*p).second;
    return (*m_pool.insert(std::make_pair(typeSize,Allocator(typeSize, m_blockSize))).first).second;
  }

  void wipe() {
    std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Allocator::wipe,
							  boost::bind(&Pool::value_type::second,_1)
							  ));
  }

  void clear() {
    std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Allocator::clear,
							  boost::bind(&Pool::value_type::second,_1)
							  ));
  }

  template<typename Visitor>
  void visit(Visitor& visitor) const {
    std::for_each(m_pool.begin(),m_pool.end(),boost::bind(&Visitor::dump,visitor,
							  boost::bind(&Pool::value_type::second,_1)
							  ));
  }


private:
  std::size_t m_blockSize;
  Pool m_pool;
};

BlockWipedPool & blockWipedPool() {
  static BlockWipedPool local(1024);
  return local;
}

template<size_t S>
BlockWipedAllocator & blockWipedAllocator() {
  static BlockWipedAllocator & local = blockWipedPool().allocator(S);
  return local;
}


/*  general Allocator
 * 
 */
class BlockWipedPoolAllocated {
public:
  static void * operator new(size_t s) {
    return allocator(s).alloc();
  }
  
  static void operator delete(void * p, size_t s) {
    // allocator(s).dealloc(p);
  }
  
  static BlockWipedAllocator & allocator(size_t s) {
    return  blockWipedPool().allocator(s);
  }
  

  static BlockWipedAllocator::Stat stat(size_t s) {
    return allocator(s).stat();
  }
  
private:
  
  // static BlockAllocator * s_allocator;
};



/*  Allocator by type
 * 
 */
template<typename T>
class BlockWipedAllocated {
public:
  static void * operator new(size_t) {
    return allocator().alloc();
  }
  
  static void operator delete(void * p) {
    allocator().dealloc(p);
  }
  
  static BlockWipedAllocator & allocator() {
    static BlockWipedAllocator & local = blockWipedPool().allocator(sizeof(T));
    return local;
  }
  

  static BlockWipedAllocator::Stat stat() {
    return allocator().stat();
  }
  
private:
  
  // static BlockAllocator * s_allocator;
};

/*  Allocator by size
 * 
 */
template<typename T>
class SizeBlockWipedAllocated {
public:
  static void * operator new(size_t) {
    return allocator().alloc();
  }
  
  static void operator delete(void * p) {
    allocator().dealloc(p);
  }
  
  static BlockWipedAllocator & allocator() {
    static BlockWipedAllocator & local = blockWipedAllocator<sizeof(T)>();
    return  local;
  }
  

  static BlockWipedAllocator::Stat stat() {
    return allocator().stat();
  }
  
private:
  
  // static BlockAllocator * s_allocator;
};


#endif // BlockAllocator_H
