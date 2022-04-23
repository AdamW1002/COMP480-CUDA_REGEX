#pragma once
#include <iostream>
#include <fstream>
#include <string>

std::string loadBook(const char* filename);
char* processBook(std::string* book, int* len);