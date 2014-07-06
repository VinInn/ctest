#ifndef PerfTools_PPCRealTime_H
#define PerfTools_PPCRealTime_H

namespace perftools {

  namespace ppc_detail {
    typedef unsigned long long int u64;
    
    inline void timeBasePPC( u64* t ) { 
      typedef unsigned int u32;
      u32 Low; 
      u32 HighBefore; 
      u32 HighAfter; 
    
      // Complete pending instructions. 
      __asm__ volatile( "isync" ); 
    
    ////     
    A:// 
    ////     
    
      // Get upper time base register. 
      __asm__ volatile( "mftbu %0" : "=r" (HighBefore) ); 
      
      // Get lower time base register. 
      __asm__ volatile( "mftb %0" : "=r" (Low) ); 
      
      // Get upper time base register. 
      __asm__ volatile( "mftbu %0" : "=r" (HighAfter) ); 
      
      // If the upper register has changed while reading 
      // the lower register. 
      if( HighBefore != HighAfter ) 
	{ 
	  // Try again. 
	  goto A; 
	} 
      
      // Return the result as a 64-bit number. 
      *t = ( ( (u64) HighAfter << 32 ) | ( (u64) Low ) ); 
    } 
    
  }
  
  inline unsigned long long int rdtsc() {
    unsigned long long int x;
    ppc_detail::timeBasePPC(&x);
    return x;
  }
  
}

#endif // PerfTools_PPCRealTime_H
