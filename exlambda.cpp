// lambda_structure.cpp
// compile with: /EHsc
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;


void exLambda0() 
{

// The number of elements in the vector.
const int elementCount = 9;


   // Create a vector object with each element set to 1.
   vector<int> v(elementCount, 1);

   // These variables hold the previous two elements of the vector.
   int x = 1;
   int y = 1;

   // Assign each element in the vector to the sum of the 
   // previous two elements.
   generate_n(v.begin() + 2, elementCount - 2, [=]() mutable throw() -> int {
      
      // Generate current value.
      int n = x + y;

      // Update previous two values.
      x = y;
      y = n;

      return n;
   });

   // Print the contents of the vector.
   for_each(v.begin(), v.end(), [](int n) { cout << n << " "; });
   cout << endl;

   // Print the local variables x and y.
   // The values of x and y hold their initial values because 
   // they are captured by value.
   cout << x << " " << y << endl;
}
// captures_lambda_expression.cpp
// compile with: /EHsc
#include <iostream>
using namespace std;

void exMutable()
{
   int m = 0, n = 0;
   [&, n] (int a) mutable { m = ++n + a; }(4);
   cout << m << endl << n << endl;
}


// lambda_static_variable.cpp
// compile with: /c /EHsc
#include <vector>
#include <algorithm>
using namespace std;


void fillVector(vector<int>& v)
{
   // A local static variable.
   static int nextValue = 1;

   // The lambda expression that appears in the following call to
   // the generate function modifies and uses the local static 
   // variable nextValue.
   generate(v.begin(), v.end(), [] { return nextValue++; });
}

// declaring_lambda_expressions1.cpp
#include <functional>

void exAssign()
{
   // Assign the lambda expression that adds two numbers to an auto variable.
   auto f1 = [] (int x, int y) { return x + y; }; 

   // Assign the same lambda expression to a function object.
   function<int (int, int)> f2 = [] (int x, int y) { return x + y; };

   // Invoke the function object and print its result.
   cout << f1(21, 12) << endl;
   cout << f2(21, 12) << endl;

}

// declaring_lambda_expressions2.cpp
// compile with: /EHsc
#include <iostream>
#include <functional>

void exAssign2()
{
   using namespace std;

   int i = 3;
   int j = 5;

   // The following lambda expression captures i by value and
   // j by reference.
   function<int (void)> f = [i, &j] { return i + j; };

   // Change the values of i and j.
   i = 22;
   j = 44;

   // Call f and print its result.
   cout << f() << endl;
}

// calling_lambda_expressions1.cpp
// compile with: /EHsc
#include <iostream>

void exCall1()
{
   using namespace std;
   int n = [] (int x, int y) { return x + y; }(5, 4);
   cout << n << endl;
}

// calling_lambda_expressions2.cpp
// compile with: /EHsc
#include <list>
#include <algorithm>
#include <iostream>

void  call2()
{
   using namespace std;

   // Create a list of integers with a few initial elements.
   list<int> numbers;
   numbers.push_back(13);
   numbers.push_back(17);
   numbers.push_back(42);
   numbers.push_back(46);
   numbers.push_back(99);

   // Use the find_if function and a lambda expression to find the 
   // first even number in the list.
   const list<int>::const_iterator result =
      find_if(numbers.begin(), numbers.end(),
         [](int n) { return (n % 2) == 0; });

   // Print the result.
   if (result != numbers.end())
   {
       cout << "The first even number in the list is " 
            << (*result) 
            << "." 
            << endl;
   }
   else
   {
       cout << "The list contains no even numbers." 
            << endl;
   }
}

// template_lambda_expression.cpp
// compile with: /EHsc
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

// Negates each element in the vector object.
template <typename T> 
void negate_all(vector<T>& v)
{
    for_each(v.begin(), v.end(), [] (T& n) { n = -n; } );
}

// Prints to the console each element in the vector object.
template <typename T> 
void print_all(const vector<T>& v)
{
   for_each(v.begin(), v.end(), [] (const T& n) { cout << n << endl; } );
}

void exTemplate()
{
   // Create a vector of integers with a few initial elements.
   vector<int> v;
   v.push_back(34);
   v.push_back(-43);
   v.push_back(56);

   // Negate each element in the vector.
   negate_all(v);

   // Print each element in the vector.
   print_all(v);
}


// higher_order_lambda_expression.cpp
// compile with: /EHsc
#include <iostream>
#include <functional>

void hoLambda()
{
   using namespace std;

   // The following code declares a lambda expression that returns 
   // another lambda expression that adds two numbers. 
   // The returned lambda expression captures parameter x by value.
   auto g = [](int x) -> function<int (int)> 
      { return [=](int y) { return x + y; }; };

   // The following code declares a lambda expression that takes another
   // lambda expression as its argument.
   // The lambda expression applies the argument z to the function f
   // and adds 1.
   auto h = [](const function<int (int)>& f, int z) 
      { return f(z) + 1; };

   // Call the lambda expression that is bound to h. 
   auto a = h(g(7), 8);

   // Print the result.
   cout << a << endl;
}
