struct Crap {
  int i;
};

namespace local {
  using ::Crap;


}



int main() {


  local::Crap c;

  return 0;
}
