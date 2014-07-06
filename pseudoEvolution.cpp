// pseduo schema evolution


// old stuff

struct WillChange {

  int i;

};

struct StillValid {

  float a;

}; 


// new stuff

namespace persistentV00 {
  using ::WillChange;
  using ::StillValid;

}

namespace dataFormat {
  using persistentV00::StillValid;


  struct WillChange {
    WillChange(persistentV00::WillChange const & old) : j(0), k(0.1*float(old.i)){}


    int j;
    float k;
  };


}

struct Event {
  dataFormat::StillValid * stillValid;
  dataFormat::WillChange * willChange;

};


void read(Event & e) {
  persistentV00::StillValid * stillValid =  new StillValid();
  persistentV00::WillChange * willChange =  new WillChange();
  e.stillValid = stillValid ;
  e.willChange = new dataFormat::WillChange(*willChange);
  
}



using namespace dataFormat;

void use(Event const & e){
  StillValid const * stillValid =  e.stillValid;
  WillChange const * willChange =  e.willChange;
}

int main() {

  Event e;
  read(e);
  use (e);
  
  return 0;
}


