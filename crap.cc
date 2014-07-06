#include <fstream>
int main () {
auto && my_read = std::fstream::in; //<== Error: expected type-specifier
std::fstream fs;
fs.open ("test.txt", my_read | std::fstream::out | std::fstream::app);
fs << " more lorem ipsum";
fs.close();
return 0;
}
