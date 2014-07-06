/*
 *  bitcompress.cpp
 *  
 *
 *  Created by Vincenzo on 29-03-2006.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */

// #include "bitcompress.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <string>

bool debug() {
	return true;
}


/*
 * generates a tag
 * assume 
 *  256 trigger-paths 20% overlap??
 *  256 offline confirmations (full correlation with above?)
 *  256 physics selections (selectivity (on confirmed) as required, 10% overlap)
 *  128 "user" bits (many on, say 10 random???)
 */
class EvTag {
public:
	void dump(int lev=0) const {
		std::cout << "sizes "
			 << triggerPaths.size() << " " 
			 << confirmations.size() << " " 
			 << selections.size() << " " 
			 << user.size() << std::endl; 	
	}

public:
	std::vector<unsigned char> triggerPaths;
	std::vector<unsigned char> confirmations;
	std::vector<unsigned char> selections;
	std::vector<unsigned char> user;
};

double frand() {
	static const double rmax = 1./double(RAND_MAX);
	return double(::rand())*rmax;
};

union FourByte {
    int i;
    unsigned char c[4];
};


class TagGenerator {
public:
	explicit TagGenerator(double isel) : selectivity(isel) {}
	void generate(EvTag& tag);
private:
	double selectivity;
};

void TagGenerator::generate(EvTag& tag) {
	FourByte r1;
	for(;;) {
		r1.i=::rand();
		tag.triggerPaths.push_back(r1.c[3]);
		tag.confirmations.push_back(r1.c[3]);
		if (r1.c[0]>25) break; // c[0]<127 -  0.2*127=25 
	}
	double psel = frand();
	if (psel>selectivity) return;
	tag.selections.push_back(r1.c[2]);
	if (psel<0.1*selectivity)  tag.selections.push_back(r1.c[1]); // aprox 10% multiple tag
	// user word: min 4, max 15 bits (out of 128) // not so uniform
	int nub = int(r1.c[0])%16; if (nub<4) nub=4;
	for (int i=0; i<nub; i+=4) {
		FourByte r2;  r2.i=::rand();
        for (int j=0; j<std::min(4,nub-i);j++) tag.user.push_back(r2.c[j]);
	}
	if (tag.user.size()!=nub) std::cout <<"error " << nub << " " << tag.user.size() << std::endl;
}

class CompactStream {
public:
	CompactStream(const std::string&iname, bool w) : file(iname.c_str(), w ?std::ios::out:std::ios::in){
		std::cout << "file is " << iname << std::endl;
	}
	operator bool () const { return file;} 
	void dump(const EvTag& tag) {
		dump(tag.triggerPaths); 
		dump(tag.confirmations); 
		dump(tag.selections); 
		dump(tag.user);
	}
	void dump(const std::vector<unsigned char>& v) {
		unsigned char s(v.size());
		file.write((char*)(&s),1);
		if (!v.empty()) file.write((char*)(&v.front()),v.size());
	}

	void read(EvTag& tag) {
		read(tag.triggerPaths); 
		read(tag.confirmations); 
		read(tag.selections); 
		read(tag.user);
	}
	void read(std::vector<unsigned char>& v) {
		unsigned char s;
		file.read((char *)(&s),1);
		if (s) {
			v.resize(s);
			file.read((char*)(&v.front()),v.size());
		}
	}	
	
private:
	std::fstream file;
};

void write(char * selchar="10");
void read(char * selchar="10");
int main(int argc, char * argv[]) {
	
	char * w ="r";
	if (argc>1) w = argv[1];
	if (w[0]=='w') {
		if (argc>2) write(argv[2]);
		else write();
	} else {
		if (argc>2) read(argv[2]);
		else read();
	}	
	return 0;
}	

void write(char * selchar) {
	// selectivity factor in %
	// default 10%
	int sf= 10;
	sf = ::atoi(selchar);
	float sel = 0.01*float(sf);
	std::cout << "selectivity will be " << sel << std::endl;

	// number of events
	size_t nev = 100000;
	CompactStream cs(std::string("compact_")+std::string(selchar),true);	
	TagGenerator gen(sel);
	for (size_t iev=0; iev<nev; iev++) 
	{
		EvTag tag;
		gen.generate(tag);
		if (iev%1000==0) tag.dump();
		cs.dump(tag);
	}

}

void read(char * selchar) {
	CompactStream cs(std::string("compact_")+std::string(selchar),false);	
	size_t iev(0);
	while (cs) {
		iev++;
		EvTag tag;
		cs.read(tag);
		if (iev%1000==1) tag.dump();

	}
	std::cout << "read " << iev << " events " << std::endl;

}