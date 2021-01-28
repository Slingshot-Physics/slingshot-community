#include <iostream>
#include <typeinfo>

struct collision_category_t
{

};

struct constraint_category_t
{

};

template <unsigned int N>
struct thing_s
{

};

int main(void)
{
   typedef unsigned int thing_t;
   typedef unsigned int thing2_t;

   std::cout << "typeinfo hash for thing_t: " << typeid(thing_t).name() << "\n";
   std::cout << "typeinfo hash for thing2_t: " << typeid(thing2_t).name() << "\n";
   std::cout << "typeinfo hash for unsigned int: " << typeid(unsigned int).name() << "\n";

   std::cout << "typeinfo hash for collision category: " << typeid(collision_category_t).name() << "\n";
   std::cout << "typeinfo hash for constraint category: " << typeid(constraint_category_t).name() << "\n";

   std::cout << "typeinfo hash for collision category: " << typeid(collision_category_t).hash_code() << "\n";
   std::cout << "typeinfo hash for constraint category: " << typeid(constraint_category_t).hash_code() << "\n";

   std::cout << "typeinfo hash for collision category: " << typeid(thing_s<2>).name() << "\n";
   std::cout << "typeinfo hash for constraint category: " << typeid(thing_s<3>).name() << "\n";

   std::cout << "typeinfo hash for collision category: " << typeid(thing_s<2>).hash_code() << "\n";
   std::cout << "typeinfo hash for constraint category: " << typeid(thing_s<3>).hash_code() << "\n";
   return 0;
}
