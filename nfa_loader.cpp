#include "nfa_loader.h"
#include <iostream>
#include <fstream>
#include <string>
nfa* loadNFAFromFile(std::string filename)
{
	std::ifstream fp;
	std::cout << "here\n";
	fp.open(filename);
	std::cout << "here2\n";
	std::string str;
	fp >> str;
	std::cout << str;
	std::string line;
	std::getline(fp, line);
	std::cout << "here3\n";
	std::cout << line << std::endl;
	while (std::getline(fp,line)) {
		std::cout << "here4\n";
		std::cout << line << std::endl;
	}
	fp.close();
	return nullptr;
}
