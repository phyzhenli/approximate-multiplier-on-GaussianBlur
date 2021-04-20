#ifndef GUARD_split
#define GUARD_split

#include <string>
#include <algorithm>

bool space(char c)
{
	return isspace(c);
}

bool not_space(char c)
{
	return !isspace(c);
}


template <class Out>
void split(const std::string& str, Out os)
{
    typedef std::string::const_iterator iter;

    iter i = str.begin();
    while (i != str.end()) {
        i = find_if(i, str.end(), not_space);

        iter j = find_if(i, str.end(), space);

        if (i != str.end())
            *os++ = std::string(i, j);
        i = j;
    }
}

#endif