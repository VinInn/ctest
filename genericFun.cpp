#include<algorithm>
#include<numeric>
#include<limits>

template<typename Ret, typename... Args>
struct genericExpression {
  virtual Ret operator()(Args ...) const =0;
};




struct vcut final : public genericExpression<bool,int,int> {
   bool operator()(int a, int b) const override{ return std::min(a,b)<1000;}
};


int main() {

  vcut cut; 
  genericExpression<bool,int,int> const * gcut = &cut;
  return (*gcut)(3,4);
}
