#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
namespace leonida::rgba4 {
using Inst=std::vector<std::uint32_t>;
inline Inst instruction(std::uint32_t op,std::initializer_list<std::uint32_t> args) {
    Inst w{(std::uint32_t(args.size()+1)<<16)|op};w.insert(w.end(),args);return w;
}
// Quantize only written color outputs at the requested locations. For a wider fallback attachment
// this reproduces four-bit UNORM writes for pipelines whose affected attachments use no blending.
inline bool quantize(Inst& words,unsigned mask) {
    if(!mask)return true;
    if(words.size()<5 || words[0]!=0x07230203)return false;
    std::vector<Inst> code,declarations;
    std::map<unsigned,unsigned> locations, pointer_type, width, scalar, variables;
    unsigned import_id=0,next=words[3],float_type=0;
    for(std::size_t at=5;at<words.size();) {
        const unsigned n=words[at]>>16,op=words[at]&65535u;
        if(!n || n>words.size()-at)return false;
        Inst i(words.begin()+at,words.begin()+at+n);at+=n;
        if(op==11 && n>=3) {
            const char* s=reinterpret_cast<const char*>(i.data()+2);const auto bytes=(n-2)*4;
            if(bytes>=13 && std::memcmp(s,"GLSL.std.450",13)==0)import_id=i[1];
        } else if(op==71 && n==4 && i[2]==30)locations[i[1]]=i[3];
        else if(op==22 && n==3 && i[2]==32){float_type=i[1];width[i[1]]=1;scalar[i[1]]=i[1];}
        else if(op==23 && n==4){width[i[1]]=i[3];scalar[i[1]]=i[2];}
        else if(op==32 && n==4 && i[2]==3)pointer_type[i[1]]=i[3];
        else if(op==59 && n>=4 && i[3]==3)variables[i[2]]=i[1];
        code.push_back(std::move(i));
    }
    if(!float_type)return false;
    std::map<unsigned,unsigned> outputs;
    for(auto [id,ptr]:variables)if(locations.count(id)&&locations[id]<8&&(mask&(1u<<locations[id]))&&pointer_type.count(ptr))
        outputs[id]=pointer_type[ptr];
    if(outputs.empty())return true;
    const bool add_import=!import_id;if(add_import)import_id=next++;
    std::map<unsigned,std::array<unsigned,4>> constants;
    auto components=[&](unsigned type) -> const std::array<unsigned,4>& {
        auto [it,inserted]=constants.try_emplace(type);
        if(!inserted)return it->second;
        constexpr std::array<float,4> vals{0.f,1.f,15.f,1.f/15.f};
        std::array<unsigned,4> base{};
        for(unsigned j=0;j<4;++j){base[j]=next++;declarations.push_back(instruction(43,{scalar[type],base[j],std::bit_cast<unsigned>(vals[j])}));}
        if(width[type]==1){it->second=base;return it->second;}
        for(unsigned j=0;j<4;++j){const auto id=next++;Inst c=instruction(44,{type,id});c.resize(3+width[type],base[j]);c[0]=(unsigned(c.size())<<16)|44u;declarations.push_back(std::move(c));it->second[j]=id;}
        return it->second;
    };
    std::vector<Inst> rewritten;bool changed=false;
    for(auto& i:code) {
        const auto op=i[0]&65535u;
        if((op==65||op==66||op==83) && i.size()>=4 && outputs.count(i[3]) && pointer_type.count(i[1]))
            outputs[i[2]]=pointer_type[i[1]];
        if(op==63 && i.size()>=3 && outputs.count(i[1]))return false;
        if(op==62 && i.size()>=3 && outputs.count(i[1])) {
            const auto type=outputs[i[1]];
            if(!width.count(type)||width[type]<1||width[type]>4||scalar[type]!=float_type)return false;
            const auto& c=components(type);
            const auto clamped=next++,scaled=next++,rounded=next++,value=next++;
            rewritten.push_back(instruction(12,{type,clamped,import_id,43,i[2],c[0],c[1]})); // GLSL FClamp
            rewritten.push_back(instruction(133,{type,scaled,clamped,c[2]}));
            rewritten.push_back(instruction(12,{type,rounded,import_id,2,scaled})); // GLSL RoundEven
            rewritten.push_back(instruction(133,{type,value,rounded,c[3]}));i[2]=value;changed=true;
        }
        rewritten.push_back(std::move(i));
    }
    if(!changed)return false;
    Inst result(words.begin(),words.begin()+5);result[3]=next;bool declared=false;
    for(const auto& i:rewritten) {
        const auto op=i[0]&65535u;
        if(op==14 && add_import){Inst imp=instruction(11,{import_id,0,0,0,0});std::memcpy(imp.data()+2,"GLSL.std.450",13);result.insert(result.end(),imp.begin(),imp.end());}
        if(op==54&&!declared){for(const auto& dec:declarations)result.insert(result.end(),dec.begin(),dec.end());declared=true;}
        result.insert(result.end(),i.begin(),i.end());
    }
    if(!declared)return false;words=std::move(result);return true;
}
}
