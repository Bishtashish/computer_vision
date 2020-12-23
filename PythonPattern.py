# best way to convert a list of integers into a string, presuming that the integers are ASCII values.
#  For instance, the list [97, 98, 99] should be converted to the string 'abc'.
#  Let's assume we want to write a function to do this.


def f1(list):
        string = ""
        for item in list:
            string = string + chr(item)
        return string


# This version performs exactly the same set of string operations as the first one,
#  but gets rid of the for loop overhead in favor of the faster, implied loop of the reduce() function.


def f2(list):
        return reduce(lambda string, item: string + chr(item), list, "")

# Sure, but it does so at the cost of a function call (the lambda function) per list item.
#  I betcha it's slower, since function call overhead in Python is bigger than for loop overhead.
# (OK, so I had already done the comparisons. f2() took 60% more time than f1(). So there :-)


def f3(list):
        string = ""
        for character in map(chr, list):
            string = string + character
        return string


# f3() clocked twice as fast as f1()! The reason that this surprised us was twofold:
#  first, it uses more storage (the result of map(chr, list) is another list of the same length);
#  second, it contains two loops instead of one (the one implied by the map() function, and the for loop).

# Of course, space versus time is a well-known trade-off, so the first one shouldn't have surprised us.
#  However, how come two loops are faster than one? Two reasons.

# First, in f1(), the built-in function chr() is looked up on every iteration,
#  while in f3() it is only looked up once (as the argument to map()).
#  This look-up is relatively expensive, I told my friend,
#  since Python's dynamic scope rules mean that it is first looked up (unsuccessfully)
#  in the current module's global dictionary, and then in the dictionary of built-in function (where it is found).
#  Worse, unsuccessful dictionary lookups are (on average) a bit slower than successful ones, because of the way the hash chaining works.

# The second reason why f3() is faster than f1() is that the call to chr(item),
#  as executed by the bytecode interpreter, is probably a bit slower than when executed
#  by the map() function - the bytecode interpreter must execute three bytecode instructions
#  for each call (load 'chr', load 'item', call), while the map() function does it all in C.


def f4(list):
        string = ""
        lchr = chr
        for item in list:
            string = string + lchr(item)
        return string

# As expected, f4() was slower than f3(), but only by 25%; it was about 40% faster than f1() still.
#  This is because local variable lookups are much faster than global or built-in variable lookups:
#  the Python "compiler" optimizes most function bodies so that for local variables,
#  no dictionary lookup is necessary, but a simple array indexing operation is sufficient.
#  The relative speed of f4() compared to f1() and f3() suggests that both reasons why f3() is faster contribute,
#  but that the first reason (fewer lookups) is a bit more important.
#  (To get more precise data on this, we would have to instrument the interpreter.)



# I was worried that the quadratic behavior of the algorithm was killing us.
#  So far, we had been using a list of 256 integers as test data,
#  since that was what my friend needed the function for.
#  But what if it were applied to a list of two thousand characters?
#  We'd be concatenating longer and longer strings, one character at a time.
#  It is easy to see that, apart from overhead, to create a list of length N in this way,
#  there are 1 + 2 + 3 + ... + (N-1) characters to be copied in total, or N*(N-1)/2, or 0.5*N**2 - 0.5*N.
#  In addition to this, there are N string allocation operations, but for sufficiently large N,
#  the term containing N**2 will take over. Indeed, for a list that's 8 times as long (2048 items),
#  these functions all take much more than 8 times as long; close to 16 times as long, in fact.
#  I didn't dare try a list of 64 times as long.

# There's a general technique to avoid quadratic behavior in algorithms like this.
#  I coded it as follows for strings of exactly 256 items:



def f5(list):
        string = ""
        for i in range(0, 256, 16): # 0, 16, 32, 48, 64, ...
            s = ""
            for character in map(chr, list[i:i+16]):
                s = s + character
            string = string + s
        return string


# Unfortunately, for a list of 256 items, this version ran a bit slower (though within 20%) of f3().
#  Since writing a general version would only slow it down more,
#  we didn't bother to pursue this path any further (except that we also compared it with a variant that didn't use map(),
#  which of course was slower again).

# Finally, I tried a radically different approach: use only implied loops.
#  Notice that the whole operation can be described as follows:
#  apply chr() to each list item;
#  then concatenate the resulting characters.
#  We were already using an implied loop for the first part: map().
#  Fortunately, there are some string concatenation functions in the string module that are implemented in C.
#  In particular, string.joinfields(list_of_strings, delimiter) concatenates a list of strings,
#  placing a delimiter of choice between each two strings.
#  Nothing stops us from concatenating a list of characters (which are just strings of length one in Python),
#  using the empty string as delimiter. Lo and behold:


import string
    def f6(list):
        return string.joinfields(map(chr, list), "")
# This function ran four to five times as fast as our fastest contender, f3().
#  Moreover, it doesn't have the quadratic behavior of the other versions.

# And The Winner Is...
# an odd corner of Python: the array module.
#  This happens to have an operation to create an array of 1-byte wide integers from a list of Python integers,
#  and every array can be written to a file or converted to a string as a binary data structure.
#  Here's our function implemented using these operations:

import array
    def f7(list):
        return array.array('B', list).tostring()

# This is about three times as fast as f6(),
#  or 12 to 15 times as fast as f3()!
#  it also uses less intermediate storage - it only allocates 2 objects of N bytes (plus fixed overhead),
#  while f6() begins by allocating a list of N items,
#  which usually costs 4N bytes (8N bytes on a 64-bit machine) - assuming the character objects are shared 
#  with similar objects elsewhere in the program (like small integers,
#  Python caches strings of length one in most cases).



# ---------------------------------------------------------------------
import time
    def timing(f, n, a):
        print f.__name__,
        r = range(n)
        t1 = time.clock()
        for i in r:
            f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a); f(a)
        t2 = time.clock()
        print round(t2-t1, 3)