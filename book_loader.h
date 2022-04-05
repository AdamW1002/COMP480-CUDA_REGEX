#pragma once
#include <iostream>
#include <fstream>
#include <string>

std::string loadBook(char* filename);
char* processBook(std::string* book, int* len);