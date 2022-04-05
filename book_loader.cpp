#include "book_loader.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

std::string loadBook(char * filename)
{
	std::ifstream file;
	file.open(filename);
	
	if (!file.is_open()) {
		std::cout << "failed to open file: " <<std::endl;
		return nullptr;
	}

	std::stringstream file_contents; //write file contents to buffer and dump to string
	file_contents << file.rdbuf();
	return file_contents.str(); //return whole string

	
}

char* processBook(std::string* book, int* len)
{
	char* buff = (char*)malloc(sizeof(char) * (book->length()));
	
	if (buff == NULL) {
		std::cout << "failed to allocate buffer" << std::endl;
		return NULL;
	}

	int count = 0;
	for (char c : *book) { //remove weird chars
		//https://www.ascii-code.com/ make sure char is a normal english char
		if (c >= 32 && c <= 126) {
			
			buff[count] = c;
			count++;
		}
	}
	buff = (char*) realloc(buff, count * sizeof(char)); //downsize since we shortened the book
	if (buff == NULL) {
		std::cout << "failed to re-allocate buffer" << std::endl;
		return NULL;
	}
	*len = count; //record length of string
	return buff;
}
