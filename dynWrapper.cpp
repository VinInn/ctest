struct BaseWapper {

  virtual ~BaseWrapper(){}

};


template<typename T>
struct Wrapper final : public BaseWrapper {

  virtual ~Wrapper(){}

  T & get() { return t;}
  T const & get() const { return t;}

  T t;
};


template<typename T>
T * get(BaseWapper & bw)  {
   return dynamic_cast<Wrapper<T> *>(&bw);
}


template<typename T>
T const * get(BaseWapper const & bw)  {
   return dynamic_cast<Wrapper<T> const *>(&bw);
}




