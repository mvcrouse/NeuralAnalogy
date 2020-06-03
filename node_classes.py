# python imports
import sys

class Node:
    #
    # Class for directed graph node 
    #

    pred_type = 'predicate'
    func_type = 'functor'
    const_type = 'constant'

    def __init__(self, label, args=None, parents=None, ordered=True, 
                 type_of=None):
        if args == None: args = []
        if parents == None: parents = []
        if type_of == None: type_of = Node.pred_type if args else Node.const_type
        self.label = label
        self.args = args
        self.parents = parents
        self.ordered = ordered
        self.type_of = type_of

    def key_form(self):
        return (self.type_of, len(self.args), self.ordered)

    def dependencies(self):
        deps = set([arg for arg in self.args])
        for arg in self.args:
            deps = deps.union(arg.dependencies())
        return deps

    def dep_subgraph(self):
        return self.dependencies().union(set([self]))

    def ancestors(self):
        anc = set(self.parents)
        for p in self.parents:
            anc = anc.union(p.ancestors())
        return anc

    def conv_isa(self):
        if self.label == 'isa':
            self.label = self.args[1].label
            self.args = [self.args[0]]
        else:
            for a in self.args:
                a.conv_isa()

    def __str__(self):
        if self.args:
            return '(' + self.label + ' ' + \
                ' '.join([str(x) for x in self.args]) + ')'
        return self.label

    def obj_id_str(self):
        if self.args:
            return '(' + self.label + str(id(self)) + ' ' + \
                ' '.join([x.obj_id_str() for x in self.args]) + ')'
        return self.label + str(id(self))

    def __repr__(self):
        return str(self)
