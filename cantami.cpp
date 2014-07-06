#include<vector>
#include<string>

std::vector<std::vector<std::wstring>> sing =
{
{L"μῆνιν", L"ἄειδε", L"θεὰ", L"Πηληϊάδεω", L"Ἀχιλῆος"},
{L"οὐλομένην,", L"ἣ", L"μυρί'", L"Ἀχαιοῖς", L"ἄλγε'", L"ἔθηκε,"},
{L"πολλὰς", L"δ'", L"ἰφθίμους", L"ψυχὰς", L"Ἄϊδι", L"προί̈αψεν"},
{L"ἡρώων,", L"αὐτοὺς", L"δὲ", L"ἑλώρια", L"τεῦχε", L"κύνεσσιν"},
{L"οἰωνοῖσί", L"τε", L"πᾶσι,", L"Διὸς", L"δ'", L"ἐτελείετο", L"βουλή"}
};

std::vector<std::string> a = {"μῆνιν", "ἄειδε", "θεὰ", "Πηληϊάδεω", "Ἀχιλῆος"};

std::vector<int> i = {0,1,2,3};

std::vector<std::string> b = { "aq", "w e", "dfr"};


std::wstring w = L"μῆνιν";

#include<iostream>
int main() {

  for (auto v : sing) {
   for (auto s: v) std::wcout <<  s <<" ";
   std::wcout <<std::endl;
  }	


   for (auto s: a) std::cout <<  s <<" ";
   std::cout <<std::endl;

  return 0;
}
