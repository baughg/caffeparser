#ifndef NNGRAPH_H
#define NNGRAPH_H 
#include <map>
#include <deque>
#include <list>

namespace movidius {    
  typedef enum
  {
    GN_NONE,
    GN_INPUT,
    GN_DATA,
    GN_CONVOLUTION,
    GN_RELU,
    GN_POOL,
    GN_INNER_PRODUCT
  }node_type;

  class graph_node
  {
  public:
    graph_node() : in_place_(false){}
    ~graph_node() {}
    void set_name(const std::string &name) { name_ = name; }
    void get_name(std::string &name) { name = name_; }
    void set_type(node_type type) { type_ = type; }
    bool add_sink(graph_node* p_sink);    
    bool add_source(graph_node* p_source);
    bool is_source(graph_node* p_node);
    bool is_sink(graph_node* p_node);  
    bool is_inplace() { return in_place_; }
    const std::list<graph_node*> &sinks();
  private:
    bool add_inplace_sink(graph_node* p_sink);
    std::string name_;
    node_type type_;
    bool in_place_;
    std::list<graph_node*> source_;
    std::list<graph_node*> sink_;
    std::list<graph_node*> inplace_sink_;
  };

  class NNGraph
  {
  public:
    NNGraph();
    ~NNGraph();
    bool generate_dot_script(const std::string &filename);
    bool add_node(const std::string &name, const std::string &type);
    bool connect(const std::string &src, const std::string &dst);
  private:       
    graph_node* get_node(const std::string &name);
    std::map<std::string, graph_node*> node_maps_;
    std::deque<graph_node*> nodes_;
  };
}
#endif

