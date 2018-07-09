// caffe_model.cpp : Defines the entry point for the console application.
//
#include "NNGraph.h"
#include <fcntl.h>

#if defined(_MSC_VER)
#include <io.h>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include "caffe.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>



using namespace google;
using namespace google::protobuf;
using namespace movidius;

void report_net(caffe::NetParameter &net_param);
void report_net_structure(caffe::NetParameter &net_param, NNGraph &graph);
void write_data(std::string filename, std::vector<float> &data);
bool read_proto_from_text_file(const char* filename, Message* proto);
bool read_blob(
  std::string output_filename,
  const caffe::BlobProto &blob);

int main(int argc, char** argv)
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 4) {
    std::cerr << "Usage:  " << argv[0] << " model.caffemodel model.prototxt mean.binaryproto" << std::endl;
    return -1;
  }
  std::string net_proto_filename = argv[2];
  std::string mean_blob_filename = argv[3];
  caffe::NetParameter net_param;
  caffe::NetParameter net_def;
  caffe::BlobProto mean_blob;

  read_proto_from_text_file(net_proto_filename.c_str(), (Message*)&net_def);

  {
    // Read the existing caffe model.
    std::fstream input(argv[1], std::ios::in | std::ios::binary);
    if (!net_param.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse caffe model." << std::endl;
      return -1;
    }
  }

  {
    // Read the mean blob.
    std::fstream input(mean_blob_filename.c_str(), std::ios::in | std::ios::binary);
    if (!mean_blob.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse mean blob file." << std::endl;
      return -1;
    }
  }

  NNGraph graph;

  printf("from: %s\n\n", argv[1]);
  report_net(net_param);
  printf("\n\nfrom: %s\n\n", net_proto_filename.c_str());
  report_net_structure(net_def, graph);
  graph.generate_dot_script("graph_nn.dot");
  std::string mean_blob_bin_filename = mean_blob_filename;
  mean_blob_bin_filename.append(".bin");
  read_blob(mean_blob_bin_filename, mean_blob);
  return 0;
}

void report_net(caffe::NetParameter &net_param)
{
  const int num_source_layers = net_param.layer_size();
  char output_filename[256];

  for (int l = 0; l < num_source_layers; ++l)
  {
    const caffe::LayerParameter& source_layer = net_param.layer(l);
    const std::string& source_layer_name = source_layer.name();

    printf("Layer %02d: %s ", l, source_layer_name.c_str());
    const int blob_count = source_layer.blobs_size();

    for (int b = 0; b < blob_count; ++b) {
      const caffe::BlobProto &blob = source_layer.blobs(b);

      sprintf(output_filename, "%s.%d.bin",
        source_layer_name.c_str(),
        b);

      read_blob(std::string(output_filename), blob);
    }

    printf("\n");
  }
}

void report_net_structure(
  caffe::NetParameter &net_param,
  NNGraph &graph)
{
  const int num_source_layers = net_param.layer_size();
  char output_filename[256];

  for (int l = 0; l < num_source_layers; ++l)
  {
    const caffe::LayerParameter& source_layer = net_param.layer(l);
    const std::string& source_layer_name = source_layer.name();
    const std::string& source_layer_type = source_layer.type();
    const int bottom_nodes = source_layer.bottom_size();
    const int top_nodes = source_layer.top_size();
    graph.add_node(source_layer_name, source_layer_type);

    for (int bn = 0; bn < bottom_nodes; ++bn){
      printf("%s -> %s, ",
        source_layer.bottom(bn).c_str(),
        source_layer_name.c_str());

      graph.connect(
        source_layer.bottom(bn),
        source_layer_name);
    }

    for (int tn = 0; tn < top_nodes; ++tn) {
      printf("%s -> %s, ", 
        source_layer_name.c_str(),
        source_layer.top(tn).c_str());

      graph.connect(
        source_layer_name,
        source_layer.top(tn));
    }    
    
    if (bottom_nodes || top_nodes)
      printf("\n");

    printf("Layer %02d: %s '%s' ", l,
      source_layer_name.c_str(),
      source_layer_type.c_str());
    const int blob_count = source_layer.blobs_size();

    if (source_layer.has_convolution_param())
    {
      const caffe::ConvolutionParameter &conv_param = source_layer.convolution_param();
      printf(" {params: o=%d, k=%d, s=%d} ",
        conv_param.num_output(),
        conv_param.kernel_size()[0],
        conv_param.stride()[0]);

    }
    for (int b = 0; b < blob_count; ++b) {
      const caffe::BlobProto &blob = source_layer.blobs(b);

      sprintf(output_filename, "%s.%d.bin",
        source_layer_name.c_str(),
        b);

      read_blob(std::string(output_filename), blob);      
    }

    printf("\n");
  }
}

void write_data(std::string filename, std::vector<float> &data)
{
  FILE* data_file = NULL;

  const size_t data_size = data.size();

  data_file = fopen(filename.c_str(), "wb");

  if (!data_file || !data_size)
    return;

  fwrite(&data[0], sizeof(float), data_size, data_file);
  fclose(data_file);
}

bool read_proto_from_text_file(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);

  if (fd == -1)
    std::cout << "File not found: " << filename << std::endl;

  io::FileInputStream* input = new io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

bool read_blob(
  std::string output_filename,
  const caffe::BlobProto &blob)
{
  if (blob.has_shape())
  {
    const caffe::BlobShape &shape = blob.shape();
    int dims = shape.dim_size();
    printf("[");

    for (int d = 0; d < dims; ++d)
    {
      printf("%u", (uint32_t)shape.dim(d));
      if (d != (dims - 1))
        printf(",");
      else
        printf("] ");
    }
  }

  const int data_size = blob.data_size();
  
  std::vector<float> data(data_size);

  for (int i = 0; i < data_size; ++i)
  {
    data[i] = blob.data(i);
  }

  // output data
  if (data_size) {
    write_data(output_filename, data);
    printf("'%s' (%d) ", output_filename.c_str(), data_size);
  }
  
  return true;
}