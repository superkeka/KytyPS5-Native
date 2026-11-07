#include "libs/libVideoDec2.cpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace Loader {
void SymbolDatabase::Add(const SymbolResolve&, uint64_t, const std::string&) {}
}
using namespace Libs::VideoDec2;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"frame-size check failed line %d: %s\n",__LINE__,#x); std::exit(2); } } while(0)
// Independent 16x16 AVC IDR from the repository's guarded Vdecsw decoding fixture.
static unsigned char intra[]={
0x00,0x00,0x00,0x01,0x09,0x10,0x00,0x00,0x00,0x01,0x67,0x64,0x10,0x0a,0xac,0xbb,
0xd0,0x80,0x00,0x00,0x03,0x00,0x80,0x00,0x00,0x03,0x01,0x02,0x00,0x00,0x00,0x01,
0x68,0xee,0x02,0x32,0xc8,0xb0,0x00,0x00,0x01,0x65,0x88,0x84,0x3a,0xfe,0xf5,0x27,
0xf8,0x14,0xd0,0x8c,0x2d};
int main(int argc,char**) {
    Videodec2DecoderConfigInfo c{};
    c.this_size=sizeof(c);c.resource_type=1;c.codec_type=1;c.profile=100;c.max_level=42;
    c.max_frame_width=1920;c.max_frame_height=1088;c.max_dpb_frame_count=-1;c.decode_input_queue_depth=4;
    Videodec2DecoderMemoryInfo m{};m.this_size=sizeof(m);
    CHECK(QueryDecoderMemoryInfo(&c,&m)==0);
    std::printf("1080p reported output buffer: %zu bytes\n",m.max_frame_buffer_size);
    if(argc>1)return 0;
    CHECK(m.max_frame_buffer_size>=2048u*1088u*3u/2u);
    CHECK(m.max_frame_buffer_size<8u*1024u*1024u);
    c.max_frame_width=c.max_frame_height=-1;
    CHECK(QueryDecoderMemoryInfo(&c,&m)==0 && m.max_frame_buffer_size>=16u*1024u*1024u);
    c.max_frame_width=c.max_frame_height=16;
    CHECK(QueryDecoderMemoryInfo(&c,&m)==0);
    std::vector<unsigned char> cpu(m.cpu_memory_size),gpu(m.gpu_memory_size),shared(m.cpu_gpu_memory_size);
    m.cpu_memory=cpu.data();m.gpu_memory=gpu.data();m.cpu_gpu_memory=shared.data();c.compute_queue=shared.data();
    Videodec2Decoder decoder=nullptr;
    auto undersized=m;undersized.max_frame_buffer_size=0;
    CHECK(CreateDecoder(&c,&undersized,&decoder)==VIDEODEC2_ERROR_MEMORY_SIZE);
    CHECK(CreateDecoder(&c,&m,&decoder)==0 && decoder);
    auto* allocation=static_cast<unsigned char*>(std::aligned_alloc(256,m.max_frame_buffer_size+512));
    CHECK(allocation);std::memset(allocation,0xa7,m.max_frame_buffer_size+512);
    Videodec2FrameBuffer fb{sizeof(fb),allocation+256,m.max_frame_buffer_size,false};
    Videodec2InputData in{sizeof(in),intra,sizeof(intra),1234,1000,0x1234};
    Videodec2OutputInfo out{};out.this_size=sizeof(out);
    CHECK(Decode(decoder,&in,&fb,&out)==0);
    if(!out.is_valid)CHECK(Flush(decoder,&fb,&out)==0);
    CHECK(out.is_valid && out.frame_width==16 && out.frame_height==16 && out.frame_pitch==256);
    CHECK(out.frame_buffer==fb.frame_buffer && out.frame_buffer_size==m.max_frame_buffer_size);
    CHECK(static_cast<unsigned char*>(out.frame_buffer)[0]==40);
    for(size_t i=0;i<256;++i)CHECK(allocation[i]==0xa7 && allocation[256+m.max_frame_buffer_size+i]==0xa7);
    CHECK(DeleteDecoder(decoder)==0);std::free(allocation);
    std::puts("bounded/unbounded sizing, minimum-size creation and guarded real AVC decode passed");
}
