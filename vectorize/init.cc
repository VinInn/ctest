#include <cstdlib>
#include <cstring>
#include<vector>

  constexpr unsigned short NOWHERE = 62001;


 class OTiledJet {
  public:
    explicit OTiledJet(int noloc) : NN(noloc), jet_index(noloc) {}
    float     eta=100.f, phi=100.f, kt2=1.e26, NN_dist=10000.f;
    unsigned short NN=62005; 
    unsigned short jet_index=62005, tile_index=NOWHERE;
    bool update=false;
    inline void label_minheap_update_needed() {update=true;}
    inline void label_minheap_update_done()   {update=false;}
    inline bool minheap_update_needed() const {return update;}
  };



OTiledJet* init(unsigned int sz,  unsigned int noloc) {
      OTiledJet* jets= ((OTiledJet*)(::malloc(sz*sizeof(OTiledJet))) );
      for (unsigned int i=0; i!=sz; ++i) jets[i]=OTiledJet(noloc);
      return jets;
}


std::vector<OTiledJet> initV(unsigned int sz,  unsigned int noloc) {
  std::vector<OTiledJet> jets(sz,OTiledJet(noloc));
  return jets;
}



void copy(OTiledJet & __restrict__ lh, OTiledJet const & __restrict__ rh ) {
   lh=rh;
}


void copy(unsigned char * __restrict__ lh, unsigned char const * __restrict__ rh, int n) {
   for (int i=0; i!=n; i++) lh[i]=rh[i];
}


  OTiledJet* minit(unsigned int sz,  unsigned int noloc) {
      OTiledJet* jets= ((OTiledJet*)(::malloc(sz*sizeof(OTiledJet))) );
      unsigned char * cj = (unsigned char *)(jets);
      auto last = cj+sz*sizeof(OTiledJet);
      auto half = cj + (last-cj)/2;
      jets[0]=OTiledJet(noloc);
      auto start = cj+sizeof(OTiledJet);
      auto bs = start-cj;
      while (start<half) {
        ::memcpy(start,cj,bs);
        start+=bs;                                                      
        bs = start-cj;
      };
      // assert(last-start<=start-cj);
      ::memcpy(start,cj,last-start);
      return jets;
  }

