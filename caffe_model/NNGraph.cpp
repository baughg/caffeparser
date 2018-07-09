#include "NNGraph.h"
#include <fstream>

using namespace movidius;

NNGraph::NNGraph()
{
}

NNGraph::~NNGraph()
{
  const size_t nodes = nodes_.size();

  for (size_t n = 0; n < nodes; ++n)
  {
    delete nodes_.front();
    nodes_.pop_front();
  }
}

bool NNGraph::add_node(const std::string &name, const std::string &type)
{
  graph_node* p_g_node = get_node(name);
  
  node_type n_type = GN_NONE;
  const char* p_type = type.c_str();

  if (!strcmp(p_type, "Convolution"))
    n_type = GN_CONVOLUTION;
  else if (!strcmp(p_type, "Input"))
    n_type = GN_INPUT;

  if (!p_g_node) {
    graph_node* p_g_node = new graph_node();
    p_g_node->set_name(name);
    p_g_node->set_type(n_type);
    node_maps_[name] = p_g_node;
    nodes_.push_back(p_g_node);
    return true;
  }
  else
  {
    p_g_node->set_type(n_type);
  }
    
  return false;
}

bool NNGraph::connect(
  const std::string &src, 
  const std::string &dst)
{
  if (strcmp(src.c_str(), dst.c_str()) == 0)
    return false;

  graph_node* p_src_node = get_node(src);
  graph_node* p_dst_node = get_node(dst);

  if (!p_src_node) {
    add_node(src, "none");
    p_src_node = get_node(src);
  }

  if (!p_dst_node) {
    add_node(dst, "none");
    p_dst_node = get_node(dst);
  }

  if (!p_dst_node || !p_src_node)
    return false;

  if(p_src_node->add_sink(p_dst_node))
    p_dst_node->add_source(p_src_node);

  return true;
}

graph_node* NNGraph::get_node(const std::string &name)
{
  std::map<std::string, graph_node*>::iterator it; 
  it = node_maps_.find(name);

  if (it == node_maps_.end()) {
    return NULL;
  }

  return it->second;
}

bool graph_node::add_sink(graph_node* p_sink)
{
  if (is_source(p_sink))
  {
    in_place_ = true;
    add_inplace_sink(p_sink);
    //return false;
  }

  std::list<graph_node*>::iterator sink;
  sink = std::find(sink_.begin(), sink_.end(), p_sink);

  if (sink == sink_.end()) {
    sink_.push_back(p_sink);
    return true;
  }

  return false;
}

bool graph_node::add_inplace_sink(graph_node* p_sink)
{
  std::list<graph_node*>::iterator sink;
  sink = std::find(inplace_sink_.begin(), inplace_sink_.end(), p_sink);

  if (sink == inplace_sink_.end()) {
    inplace_sink_.push_back(p_sink);
    return true;
  }

  return false;
}

bool graph_node::is_source(graph_node* p_node)
{
  std::list<graph_node*>::iterator source;
  source = std::find(source_.begin(), source_.end(), p_node);

  return source != source_.end();
}

bool graph_node::is_sink(graph_node* p_node)
{
  std::list<graph_node*>::iterator sink;
  sink = std::find(sink_.begin(), sink_.end(), p_node);

  return sink != sink_.end();
}

bool graph_node::add_source(graph_node* p_source)
{
  std::list<graph_node*>::iterator source;
  source = std::find(source_.begin(), source_.end(), p_source);

  if (source == source_.end()) {
    source_.push_back(p_source);
    return true;
  }

  return false;
}

bool NNGraph::generate_dot_script(const std::string &filename)
{
  std::ofstream nn_graph(filename.c_str());

  nn_graph << "digraph G{\n" << std::endl;
  nn_graph << "\tgraph[bgcolor = white,fontname=Courier,fontsize=8.0,labeljust=l,nojustify=true]" << std::endl;
  nn_graph << "\tnode[shape = box fillcolor = cyan style = \"filled\" fontcolor = black]" << std::endl;
  nn_graph << "\tedge[color = black]" << std::endl;

  /*nn_graph
    << "\t" << p_function->safe_name
    << " [label=\""
    << p_function->name
    << "\"]"
    << std::endl;

  sprintf(colour_code_str, "%.6X", p_function->colour);

  frame_graph
    << "\t" << p_function->safe_name
    << " [fillcolor=\"#"
    << colour_code_str
    << "\"]" << std::endl;

  frame_graph << "\t" << processor_name << "->" << p_function->safe_name << std::endl;*/

  const size_t nodes = nodes_.size();
  std::string source_node_name;
  std::string sink_node_name;
  std::string node_label;

  for (size_t n = 0; n < nodes; ++n)
  {
    graph_node* p_node = nodes_.front();
    nodes_.pop_front();
    nodes_.push_back(p_node);

    if (p_node->is_inplace())
      continue;

    p_node->get_name(source_node_name);
    node_label = "<B>";
    node_label.append(source_node_name);
    node_label.append("</B><BR/>");

    std::list<graph_node*> sinks = p_node->sinks();
    
    for (std::list<graph_node*>::iterator s = sinks.begin(); 
      s != sinks.end(); ++s)
    {
      (*s)->get_name(sink_node_name);
      if (!(*s)->is_inplace())
      {
        nn_graph
          << "\t" << source_node_name.c_str()
          << "->"
          << sink_node_name.c_str()
          << std::endl;
      }
      else
      {
        node_label.append(": ");
        node_label.append(sink_node_name);
        node_label.append("<BR align=\"left\"/>");
      }
    }

    node_label.append("\n");
    nn_graph
      << "\t" << source_node_name.c_str()
      << " [label=<"
      << node_label.c_str()
      << ">]"
      << std::endl;
  }

  nn_graph << "}" << std::endl;
  nn_graph.close();
  return true;
}

const std::list<graph_node*> &graph_node::sinks()
{
  return sink_;
}