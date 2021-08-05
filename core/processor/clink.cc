#include "clink.h"

#include "core/common/common.h"
#include "core/common/sample_list.h"
#include "core/processor/clink_impl.h"
#include "core/utils/feature_internal.h"
namespace clink {

Clink::Clink() { clink_impl_ = std::make_shared<ClinkImpl>(); }

int Clink::LoadConfig(const std::string &config_path) {
  return clink_impl_->LoadConfig(config_path);
}

int Clink::LoadConfig(const std::string &remote_url,
                      const std::string &config_path) {
  return clink_impl_->LoadConfig(remote_url, config_path);
}

int Clink::FeatureExtract(const Sample &sample, std::vector<uint32_t> *index,
                          std::vector<float> *value) {
  return clink_impl_->FeatureExtract<Sample>(sample, index, value);
}

int Clink::FeatureExtract(const std::string &input,
                          std::vector<uint32_t> *index,
                          std::vector<float> *value) {
  return clink_impl_->FeatureExtract<std::string>(input, index, value);
}

int Clink::FeatureExtract(const SampleRecord &input,
                          std::vector<uint32_t> *index,
                          std::vector<float> *value) {
  return clink_impl_->FeatureExtract<SampleRecord>(input, index, value);
}

extern "C" FEATURE_DLL_DECL Clink *load_plugin(void) { return new Clink; }

extern "C" FEATURE_DLL_DECL void destroy_plugin(Clink *p) { delete p; }

static clink::ClinkImpl *processor = nullptr;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void InitClink() { processor = new clink::ClinkImpl(); }

extern "C" FEATURE_DLL_DECL int FeatureOfflineInit(const char *remote_url,
                                                   const char *local_path) {
  pthread_once(&once, InitClink);
  if (processor == nullptr) {
    std::cerr << "init clink failed" << std::endl;
    return ERR_NOT_INIT;
  }
  int res = processor->LoadConfig(remote_url, local_path);
  if (res != STATUS_OK) {
    delete processor;
    processor = nullptr;
    std::cerr << "init clink failed" << std::endl;
    return res;
  }
  return STATUS_OK;
}

extern "C" FEATURE_DLL_DECL int FeatureExtractOffline(const char *input,
                                                      char **output) {
  if (processor == nullptr) {
    std::cerr << "clink not init" << std::endl;
    return ERR_NOT_INIT;
  }

  std::vector<uint32_t> index;
  std::vector<float> value;
  std::string in(input);
  int result = processor->FeatureExtract(in, &index, &value);

  for (int i = 0; i < index.size(); i++) {
    std::cout << index.at(i) << ":" << value.at(i) << " ";
  }
  std::cout << std::endl;

  if (result != clink::STATUS_OK) return result;
  std::ostringstream out;
  if (index.size() != value.size()) {
    return clink::ERR_INDEX_VALUE_UNEQUAL;
  }
  for (int i = 0; i < index.size(); ++i) {
    out << index.at(i) << ":" << value.at(i) << " ";
  }
  *output = reinterpret_cast<char *>(malloc(out.str().size() + 1));
  strcpy(*output, out.str().c_str());
  return result;
}

extern "C" FEATURE_DLL_DECL int FeatureOfflineCleanUp(char *output) {
  if (output) {
    free(output);
  }
  return 0;
}

}  // namespace clink