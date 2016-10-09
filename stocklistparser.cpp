#include <iostream>
#include <string>
using namespace std;

int main() {
  int i;
  string s;
  bool column_name = 1;

  while(getline(cin, s)) {
    for(i=0;i<s.length();++i) {
      if(s[i]==',') break;
    }
    if(!column_name) {
      cout<<s.substr(0,i)<<endl;
    }
    else {
      column_name = 0;
    }
  }

  return 0;
}
