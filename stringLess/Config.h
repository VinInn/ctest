#ifndef Config_H
#define Config_H
#include "constHash64.h"
#include <unordered_map>
#include <cstring>
class Config {
public:
  typedef unsigned long long Key;
  typedef std::unordered_map<Key,void *> Registry;

  void * get(Key k) {
    auto p = m_registry.find(k);
    if (p==m_registry.end()) return 0;
    return (*p).second;
  }

  void add(Key key, void * p) {
    m_registry[key] = p;
  }
  
  Registry m_registry;

}; 

struct Registerer {
  typedef unsigned long long Key;
  static constexpr Key hash(const char * name) { return ::hash64(name,strlen(name),0);}

  Registerer(const char * obj, const char * name, void * conf) {
    add(hash(obj), hash(name),conf);
  }

  Registerer(Key obj, Key name, void * conf) {
    add(obj, name, conf);
  }

  static void add(Key obj, Key name, void * conf);

};


#define REGISTER(type,name,config) \
  namespace  local ## __LINE__ { \
    constexpr  Registerer::Key cName =  Registerer::hash(#name); \
    constexpr  Registerer::Key cType =  Registerer::hash(#type); \
    static Registerer  rc0(cType, cName, &config); \
  }


#endif
