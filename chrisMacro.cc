#define ONLY_REGISTER_PLUGIN(record_,type_)\
typedef DataProxy<record_, type_> EDM_PLUGIN_SYM(Proxy , __LINE__ ); \
DEFINE_EDM_PLUGIN( cond::ProxyFactory, EDM_PLUGIN_SYM(Proxy , __LINE__ ), #record_ "@" #type_ "@Proxy")


ONLY_REGISTER_PLUGIN("record",cond::Type);


