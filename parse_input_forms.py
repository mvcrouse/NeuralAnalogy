# python imports
import re, copy
# code imports
from node_classes import *
from utilities import *

unord_preds = ['and', 'or', 'equal', '=', 'AND', 'OR', 'EQUAL', 
              'elementsIntersect', 'edgesPerpendicular',
              'elementsConnected', 'edgesDisconnected', 'edgesParallel']
unord_funcs = ['+', '-', '*']
ord_funcs = []
unord_syms = set(unord_preds + unord_funcs)
all_funcs = set(unord_funcs + ord_funcs)
appl_node = '_EVAL_'
arg_ext_base = '_arg_' 

def parse_s_expr_to_gr(sexpr_str, conv_isa=False, max_arity=False):
    toks = re.split('([()])', sexpr_str)
    toks = [x for x in toks if x]
    stack, add_lst = [], []
    for tok in toks:
        if tok == '(':
            stack.append(add_lst)
            add_lst = []
        elif tok == ')':
            assert len(stack) > 0, 'Imbalanced parentheses:\n' + sexpr_str
            assert add_lst, 'Empty list found:\n' + sexpr_str
            old_lst = add_lst
            old_expr = convert_lst_to_node(old_lst, max_arity)
            add_lst = stack.pop()
            add_lst.append(old_expr)
        else:
            add_lst.extend(get_syms(tok))
    assert len(stack) == 0, 'Imbalanced parentheses:\n' + sexpr_str
    if conv_isa: 
        for a in add_lst: a.conv_isa()
    return coalesce_graph(add_lst)

def get_syms(string):
    return [Node(x) for x in string.split() if x]

def convert_lst_to_node(lst, max_arity):
    # if not constant
    if lst[0].args: return Node(appl_node, lst, type_of=Node.pred_type)
    elif max_arity != False and max_arity < len(lst) - 1:
        is_ord = (not lst[0].label in unord_syms)
        if probably_func_label(lst[0].label) and \
           all(a.type_of in [Node.func_type, Node.const_type] for a in lst[1:]):
            type_of = Node.func_type
        else: type_of = Node.pred_type
        lead_node = Node(lst[0].label, ordered=is_ord, type_of=type_of)
        curr_node = lead_node
        curr_args, args_at = [], 0
        for arg in lst[1:]:
            if len(curr_args) + 1 >= max_arity:
                curr_node.args = curr_args
                curr_args, args_at = [], args_at + 1
                new_node = Node(lst[0].label + arg_ext_base + str(args_at), 
                                ordered=is_ord, type_of=type_of)
                curr_node.args.append(new_node)
                curr_node = new_node
            curr_args.append(arg)
        if curr_args: curr_node.args = curr_args
        return lead_node
    else: 
        is_ord = (not lst[0].label in unord_syms)
        if not lst[1:]: type_of = Node.const_type
        elif probably_func_label(lst[0].label) and \
             all(a.type_of in [Node.func_type, Node.const_type] for a in lst[1:]):
            type_of = Node.func_type
        else: type_of = Node.pred_type
        return Node(lst[0].label, lst[1:], ordered=is_ord, type_of=type_of)

def probably_func_label(label):
    return 'Fn' in label or label in all_funcs or label[-2:].lower() == 'fn'

def revert_arg_ext(lst, w_pos=False):
    ret_lst = []
    curr_at = 0
    for i in range(len(lst)):
        node = lst[i]
        if arg_ext_base in node.label:
            for p in node.parents:
                if node in p.args: p.args.remove(node)
                p.args.extend(node.args)
                for arg in node.args:
                    if node in arg.parents: arg.parents.remove(node)
                    arg.parents.append(p)
        else:
            if w_pos:
                ret_lst.append((i, curr_at, node))
                curr_at += 1
            else: ret_lst.append(node)
    return ret_lst

if __name__ == '__main__':
    inp_str = '((GroupFn a) b)'
    res = parse_s_expr_to_gr(inp_str, max_arity=3)
    inp_str = '(I went (to the (store) )) (I store (store a)) (to the (store))'
    res = parse_s_expr_to_gr(inp_str, max_arity=3)
    print(res)
    inp_str = '((I went (to the (store ) )) (a) ()'
    res = parse_s_expr_to_gr(inp_str)
    print(res)
