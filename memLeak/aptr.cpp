#include <memory>
#include <string>
#include <iostream>
#include <vector>

#include "MemorySize.h"

    class pop {
    private:
      double foo[1000];
    };

class iggy {
public:
  iggy() {}
    void makeiggy(){ pop* weasel = new(pop);
      bowie.push_back(weasel);
    }
    void clean() {
      for (unsigned i=bowie.size(); i!=0;)
	{ if ((--i % 10000) == 0) { std::cout<<i<<" "<<MS.getValue()<<" "<<MS.getRSSValue()<<std::endl;}
	delete bowie[i];
	}
      bowie.clear();
    }
    private:
  std::vector<pop*> bowie;
  MemorySize MS;
};

int main()
  { iggy bar;
    MemorySize MS;

    for (unsigned i=0; i<1000000; ++i)
      { if (i%10000 == 0) { std::cout<<i<<" "<<MS.getValue()<<" "<<MS.getRSSValue()<<std::endl;}
	bar.makeiggy();
      }
    bar.clean();
    std::cout<<MS.getValue()<<" "<<MS.getRSSValue()<<std::endl;
    return 0;
  }
