''' Wordvector class(with attached unittests) '''
import math
import re
__all__ = ['Wordvector']

class Wordvector(dict) :
    ''' A class for creating and maniplating context word vectors. '''

    blacklist = ['', 'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 
        'you', 'that', 'he', 'was', 'for', 'on', 'are', 'with', 
        'as', 'i', 'can', 'they', 'be', 'at', 'one', 'have', 'this', 
        'from', 'or', 'had', 'by', 'but', 'some', 'what', 'there', 'we', 
        'can', 'out', 'other', 'were', 'all', 'your', 'when', 'up', 'use', 'word',
        'how', 'said', 'an', 'she', 'which', 'do', 'their', 'time', 'if', 'will', 
        'way', 'about', 'many', 'then', 'them', 'would', 'write', 'like', 'so',
        'these', 'her', 'long', 'make', 'thing', 'him', 'see', 'two', 
        'has', 'look', 'more']

    def __init__(self) :
        dict.__init__(self)

    @classmethod
    def by_dict(cls, vector) :
        ''' Create a new Wordvector '''
        if not isinstance(vector, dict) :
            raise TypeError("vector must be a dictionary")
        d = Wordvector()
        d.update(vector)
        return d

    @classmethod
    def by_text(cls, text, blacklist=list()) :
        ''' Create a Wordvector based on a corpus of text, 
            instead of a vector. '''
        if text :
            text, num_replaces = re.subn(r'[^ \w\s:]', ' ', text)
            array = text.lower().split()
            vector = dict((word, array.count(word)) 
                for word in array if word not in blacklist)
            return cls.by_dict(vector)

    @staticmethod
    def get_key_name(raw) :
        ''' get a unique key name for the vector for any given word. '''
        raw, num_replaces = re.subn(r'[^\w]+', '_', raw)
        stripped = raw.strip('_')
        return stripped.lower()

    @classmethod
    def get_comparable(cls, v1, v2) :
        ''' Returns two vectors each with the union of the keys of the original 
            two vectors, and values preserved from the originals. '''
        if not isinstance(v1, cls) or not isinstance(v2, cls) :
            raise TypeError('items must both be Wordvectors')
        _v1 = v1.union(v2)
        _v2 = v2.union(v1)
        return(_v1, _v2)

    def is_comparable(self, v) :
        ''' Returns True or raises an error if the two vectors 
            are not comparable. '''
        for word in v.keys() :    
            if word not in self.keys() :
                raise ValueError('Vectors are not comparable.')
        for word in self.keys() :
            if word not in v.keys() :
                raise ValueError('Vectors are not comparable.')
        return True

    # filters, map

    def map(self, callback) :
        ''' Map a function unto all values of the vector. '''
        return Wordvector.by_dict(dict((word, callback(val)) 
            for word, val in self.items()))

    def scalar(self, num) :
        ''' Map a function unto all values of the vector. '''
        return Wordvector.by_dict(dict((word, (val * num)) 
            for word, val in self.items()))

    def filter(self, callback) :
        ''' Filter items in vector based on value. '''
        return Wordvector.by_dict(dict((word, val) 
            for word, val in self.items() if callback(val)))
        
    def filter_on_keys(self, callback) :
        ''' Filter items of vector based on key. '''
        return Wordvector.by_dict(dict((word, val) 
            for word, val in self.items() if callback(word)))
    
    def filter_on_items(self, callback) :
        ''' Filter items of vector based on item. '''
        return Wordvector.by_dict(dict((word, val) 
            for word, val in self.items() 
            if callback((word, val))) 
       )
    
    def __setitem__(self, key, value) :
        ''' Append a new key/value pair onto the vector. '''
        if not isinstance(value, (int, float)) :
            raise ValueError("cannot append non-numerical value: %s" % str(value))
        dict.__setitem__(self, key, value)

    # auxilary filter/map functions

    def filter_by_freq(self, num) :
        ''' Filter items out of vector if value below num. '''
        return self.filter(lambda x : x >= num)

    def intersect(self, v) :
        ''' Returns a Wordvector of the key intersection. '''
        return self.filter_on_keys(lambda x : x in v.keys())

    def union(self, v) :
        ''' Returns a vector that has a union of the 
            keys of self and v. '''
        keys = self.keys()
        keys.extend(v.keys())
        return Wordvector.by_dict(dict((word, self.get(word, 0)) 
            for word in keys))

    def len(self) :
        '''  Returns the length of the vector. '''
        return math.sqrt(math.fsum(self.map(lambda x : x**2).values())) 

    def normalized(self) :
        ''' Returns a normalized version of the vector:
            i.e all values divided by the self.len(). '''
        length = self.len()
        return Wordvector.by_dict(dict((word, float(val)/float(length))
            for word, val in self.items()))

    def operate(self, op, v) :
        ''' operates on two corresponding values of two 
            comparable vectors. '''
        self.is_comparable(v)
        return Wordvector.by_dict(dict(
           (word, op(self.get(word, 0), v.get(word, 0))) 
            for word in self.keys()))
                
    def add(self, v) :
        ''' Itemwise addition. '''
        return Wordvector.by_dict(dict(
           (word, self.get(word, 0) + v.get(word, 0)) 
            for word in v.keys() + self.keys()))
    
    def sub(self, v) :
        ''' Itemwise subtraction. '''
        return Wordvector.by_dict(dict(
           (word, self.get(word, 0) - v.get(word, 0)) 
            for word in v.keys() + self.keys()))

    def div(self, v) :
        ''' Itemwise division. '''
        return Wordvector.by_dict(dict(
           (word, float(self.get(word, 0)) / float(v.get(word, 1))) 
            for word in self.keys() + v.keys()))
    
    def mul(self, v) :
        ''' Itemwise multiplication. '''
        return Wordvector.by_dict(dict(
           (word, self.get(word, 0) * v.get(word, 0)) 
            for word in self.keys() + v.keys()))

    def dot(self, v) :
        ''' Returns the dot product of the two vectors. '''
        return math.fsum(self.mul(v).values())

    def theta(self, v) : 
        ''' Returns the angle between the two vectors in radians. '''
        return math.acos(self.normalized().dot(v.normalized()))

import unittest

class TestWordvector(unittest.TestCase) :
    ''' Test class '''
 
    def setUp(self) :
        ''' setup '''
        self.v1 = Wordvector.by_dict({'hello':1, 'world':1})
        self.v2 = Wordvector.by_dict({'hello':1, 'dave':2})

    def testvector(self) :
        ''' test! '''
        self.assert_(1 == self.v1.get('hello'))
        self.assert_(1 == self.v1.get('world'))

    def testbytext(self) :
        ''' test! '''
        newvector = Wordvector.by_text('Hello World')
        self.assert_(1 == newvector.get('hello'))
        self.assert_(1 == newvector.get('world'))
        
    def testgetkeyname(self) :
        ''' test! '''
        self.assertEquals(
            'this',  
            Wordvector.get_key_name('This') 
       )
        self.assertEquals(
            'hello_world', 
            Wordvector.get_key_name('Hello World') 
       )
        self.assertEquals(
            'hello_world', 
            Wordvector.get_key_name('Hello*World') 
       )
        self.assertEquals(
            'hello_world', 
            Wordvector.get_key_name('Hello**********World') 
       )
        self.assertEquals(
            'hello_world', 
            Wordvector.get_key_name('Hello *** **** *** World') 
        )

    def testcomparable(self) :
        ''' test! '''
        c1, c2 = Wordvector.get_comparable(self.v1, self.v2)
        for key in self.v1.keys() + self.v2.keys() :
            self.assert_(key in c1.keys())
            self.assert_(key in c2.keys())

        try :
            self.assertRaises(ValueError, self.v1.is_comparable(self.v2))
            self.fail()
        except ValueError :
            pass
        try :
            self.assertRaises(ValueError, self.v2.is_comparable(self.v1))
            self.fail()
        except ValueError : 
            pass

        self.assert_(self.v2.is_comparable(self.v2))
        self.assert_(self.v1.is_comparable(self.v1))
        self.assert_(c1.is_comparable(c2))

    def testkeys(self) :
        ''' test! '''
        self.assert_(isinstance(self.v1.keys(), list))
        for key in 'hello', 'world' :
            self.assert_(key in self.v1.keys())

    def testvalues(self) :
        ''' test! '''
        self.assert_(isinstance(self.v1.values(), list))

    def testitems(self) :
        ''' test! '''
        self.assert_(isinstance(self.v1.items(), list))
        for item in self.v1.items() :
            self.assert_(isinstance(item, tuple))

    def testmap(self) :
        ''' test! '''
        m1 = self.v1.map(lambda x : 2)
        for val in m1.values() :
            self.assert_(2 == val)

        m2 = m1.map(lambda x : 2*x)
        for val in m2.values() :
            self.assert_(4 == val)

    def testfilter(self) :
        ''' test! '''
        f1 = self.v2.filter(lambda x : x == 1)
        self.assert_('hello' in f1.keys())
        self.assert_('dave' not in f1.keys())

        f2 = self.v2.filter(lambda x : x == 2)
        self.assert_('dave' in f2.keys())
        self.assert_('hello' not in f2.keys())

    def testfilteronkeys(self) :
        ''' test! '''
        self.assert_('hello' in self.v1.intersect(self.v2).keys())
        self.assert_('world' not in self.v1.intersect(self.v2).keys())

    def testlen(self) :
        ''' test! '''
        self.assertAlmostEqual(math.sqrt(2), self.v1.len())
        self.assertAlmostEqual(math.sqrt(5), self.v2.len())

        v3 = Wordvector.by_dict({'hello':3, 'world':4})
        self.assertEqual(5, v3.len())

    def testnormalized(self) :
        ''' test! '''
        self.assertAlmostEqual(
           (1.0/math.sqrt(2)), 
            self.v1.normalized().get('hello') 
       )

    def testoperate(self) :
        ''' test! '''
        self.assertEqual(
            self.v1.get('hello') + 
            self.v1.get('hello'), 
            self.v1.add(self.v1).get('hello')
       )    
        self.assertEqual(
            self.v1.get('hello') - 
            self.v1.get('hello'), 
            self.v1.sub(self.v1).get('hello')
       )    
        try :
            self.assertRaises(ValueError, self.v1.add(self.v2))
            self.fail()
        except ValueError : 
            pass

    def testdot(self) :
        ''' test! '''
        c1, c2 = Wordvector.get_comparable(self.v1, self.v2)
        self.assert_(1 == c1.dot(c2))

    def testtheta(self) :
        ''' test! '''
        c1, c2 = Wordvector.get_comparable(self.v1, self.v2)
        self.assert_(c2.theta(c1) == c1.theta(c2))
        
    
if __name__ == "__main__" :
    unittest.main()
