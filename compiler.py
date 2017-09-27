import ast
import astor
import inspect
import itertools
import operator

########
## IR ##
########
"""
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)
"""

## Exprs ##
class BinOp(ast.AST):
    _fields = ['op', 'left', 'right']

class CmpOp(ast.AST):
    _fields = ['op', 'left', 'right']

class UnOp(ast.AST):
    _fields = ['op', 'e']

class Ref(ast.AST):
    _fields = ['name', 'index']

    # def __init__(self, name, index=None):
        # return super(self, Ref).__init__(name, index)

class IntConst(ast.AST):
    _fields = ['val',]

class FloatConst(ast.AST):
    _fields = ['val',]

## Ops ##
class Add(ast.AST):
    pass

class Sub(ast.AST):
    pass

class Mul(ast.AST):
    pass

class Div(ast.AST):
    pass

class Mod(ast.AST):
    pass

class And(ast.AST):
    pass

class Or(ast.AST):
    pass

class Neg(ast.AST):
    pass

class Not(ast.AST):
    pass

class EQ(ast.AST):
    pass

class NE(ast.AST):
    pass

class LT(ast.AST):
    pass

class GT(ast.AST):
    pass

class LE(ast.AST):
    pass

class GE(ast.AST):
    pass

## Stmts ##
class Assign(ast.AST):
    _fields = ['ref', 'val']

class Block(ast.AST):
    _fields = ['body',]

class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']
    
    # def __init__(self, cond, body, elseBody=None):
        # return super(self, If).__init__(cond, body, elseBody)

class For(ast.AST):
    _fields = ['var', 'min', 'max', 'body']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body']


BIN_OP_MAP = {
    ast.Add: Add,
    ast.Sub: Sub,
    ast.Mult: Mul,
    ast.Div: Div,
    ast.Mod: Mod,
    ast.And: And,
    ast.Or: Or,
}

BIN_OP_INTERPRETER_MAP = {
    Add: operator.add,
    Sub: operator.sub,
    Mul: operator.mul,
    Div: operator.truediv,
    Mod: operator.mod,
    And: lambda x, y: x and y,
    Or: lambda x, y: x or y,
}

UN_OP_MAP = {
    ast.USub: Neg,
    ast.Not: Not,
}

UN_OP_INTERPRETER_MAP = {
    Neg: operator.neg,
    Not: operator.not_,
}

CMP_OP_MAP = {
    ast.Eq: EQ,
    ast.NotEq: NE,
    ast.Lt: LT,
    ast.Gt: GT,
    ast.LtE: LE,
    ast.GtE: GE,
}

CMP_OP_INTERPRETER_MAP = {
    EQ: operator.eq,
    NE: operator.ne,
    LT: operator.lt,
    GT: operator.gt,
    LE: operator.le,
    GE: operator.ge,
}


class PythonToSimple(ast.NodeVisitor):
    """
    Translate a Python AST to our simplified IR.
    
    As a bonus, try implementing logic to catch non-representable 
    Python AST constructs and raise a `NotImplementedError` when you
    do. Without this, the compiler will just explode with an 
    arbitrary error or generate malformed results. Carefully catching
    and reporting errors is one of the most challenging parts of 
    building a user-friendly compiler.
    """

    def visit_BinOp(self, node):
        if node.op.__class__ not in BIN_OP_MAP:
            raise NotImplementedError("{} Binary Operator not implemented".format(
                node.op.__class__.__name__))
        op = BIN_OP_MAP[node.op.__class__]()
        return BinOp(op=op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_BoolOp(self, node):
        assert(len(node.values) >= 2)
        if node.op.__class__ not in BIN_OP_MAP:
            raise NotImplementedError("{} Binary Operator not implemented".format(
                node.op.__class__.__name__))
        op = BIN_OP_MAP[node.op.__class__]()

        # TODO: Could be optimized to encourage short-circuiting or parallelization.
        last_node = self.visit(node.values[0])
        for i in range(1, len(node.values)):
            last_node = BinOp(op=op, left=last_node, right=self.visit(node.values[i]))
        return last_node

    def visit_Compare(self, node):
        assert(len(node.ops) == len(node.comparators))
        assert(len(node.ops) >= 1)

        for op in node.ops:
            if op.__class__ not in CMP_OP_MAP:
                raise NotImplementedError("{} Comparitive Operator not implemented".format(
                op.__class__.__name__))

        last_node = CmpOp(
            op=CMP_OP_MAP[node.ops[0].__class__](),
            left=self.visit(node.left),
            right=self.visit(node.comparators[0]))
        # TODO: Could be optimized to encourage parallelization.
        for i in range(1, len(node.ops)):
            op = CMP_OP_MAP[node.ops[i].__class__]()
            # Revisit the right side of the last comp op to avoid weird issues 
            # in case anything ever ends up modifying node objects.
            left = self.visit(node.comparators[i-1])
            right = self.visit(node.comparators[i])
            cmp_op = CmpOp(op=op, left=left, right=right)
            last_node = BinOp(op=And(), left=last_node, right=cmp_op)
        return last_node

    def visit_UnaryOp(self, node):
        if node.op.__class__ not in UN_OP_MAP:
            raise NotImplementedError("{} Unary Operator not implemented".format(
                node.op.__class__.__name__))
        op = UN_OP_MAP[node.op.__class__]()
        return UnOp(op=op, e=self.visit(node.operand))

    def visit_Subscript(self, node):
        if node.value.__class__ != ast.Name:
            raise NotImplementedError("Subscripting only supported for named variables")
        if node.slice.__class__ != ast.Index:
            raise NotImplementedError("Subscripting only supported for simple indexing")
        return Ref(name=node.value.id, index=self.visit(node.slice.value))

    def visit_Name(self, node):
        return Ref(name=node.id)

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(val=node.n)
        elif isinstance(node.n, float):
            return FloatConst(val=node.n)
        else:
            raise NotImplementedError("{} not implemented".fomrat(type(node.n)))
  
    def visit_Assign(self, node):
        if len(node.targets) > 1:
            raise NotImplementedError(
                "x=y=3 style assignments not supported. "
                "Only single assignemnts are supported.")
        if node.targets[0].__class__ != ast.Name:
            raise NotImplementedError(
                "x,y = 1,2 style assignments not supported. "
                "Only single assignments are supported.")
        # This works because node.targets[0] is always an ast.Name class.
        return Assign(ref=self.visit(node.targets[0]), val=self.visit(node.value))

    def visit_AugAssign(self, node):
        if node.target.__class__ != ast.Name and node.target.__class__ != ast.Subscript:
           raise NotImplementedError(
                "Left-hand of augmented assignments must be a simple variable")
        if node.op.__class__ not in BIN_OP_MAP:
            raise NotImplementedError("{} Binary Operator not implemented".format(
                node.op.__class__.__name__))
        op = BIN_OP_MAP[node.op.__class__]()
        bin_op_node = BinOp(op=op, left=self.visit(node.target), right=self.visit(node.value))
        return Assign(ref=self.visit(node.target), val=bin_op_node)

    def visit_If(self, node):
        expr = self.visit(node.test)
        body = Block([self.visit(stmt) for stmt in node.body])
        if_node = If(cond=expr, body=body) 
        if len(node.orelse) > 0:
            if_node.elseBody = Block([self.visit(stmt) for stmt in node.orelse])
        return if_node

    def visit_For(self, node):
        if len(node.orelse) > 0:
            raise NotImplementedError("For Else blocks not implemented")
        if node.target.__class__ != ast.Name:
            raise NotImplementedError("Only simple names can be the iterator for for loops")
        var = node.target.id
        if (node.iter.__class__ != ast.Call or
            node.iter.func.__class__ != ast.Name or
            node.iter.func.id != 'range' or
            len(node.iter.args) < 1 or
            len(node.iter.args) > 2):
            # We don't check the type of the args. It's valid for the range to be based
            # on a named variable, but at this point there's no way to check the value of that.
            # Instead, there will be an additional check in our custom interpreter.
            raise NotImplementedError("Only range(x) and range(x,y) are supported for the iterator")
        if len(node.iter.args) == 1:
            range_min = IntConst(val=0)
            range_max = self.visit(node.iter.args[0])
        else:
            range_min = self.visit(node.iter.args[0])
            range_max = self.visit(node.iter.args[1])
        body = Block([self.visit(stmt) for stmt in node.body])

        return For(var=var, min=range_min, max=range_max, body=body)

    def visit_Return(self, node):
        return Return(val=self.visit(node.value))
    
    def visit_FunctionDef(self, func):
        if (func.args.vararg or func.args.kwonlyargs or func.args.kw_defaults or
                func.args.kwarg or func.args.defaults):
            raise NotImplementedError("Only simple, unnamed arguments are supported")

        assert isinstance(func.body, list)
        body = Block([self.visit(stmt) for stmt in func.body])
        args = [arg.arg for arg in func.args.args]

        return FuncDef(name=func.name, args=args, body=body)

    def generic_visit(self, node):
        raise NotImplementedError("TODO: implement {}".format(node.__class__.__name__))

def Interpret(ir, *args):
    assert isinstance(ir, FuncDef)
    
    # Initialize a symbol table, to store variable => value bindings
    syms = dict(zip(ir.args, args))

    
    # Build a visitor to evaluate Exprs, using the symbol table to look up
    # variable definitions
    class EvalExpr(ast.NodeVisitor):
        def __init__(self, symbolTable):
            self.syms = symbolTable
        
        def visit_IntConst(self, node):
            return node.val

        def visit_Ref(self, node):
            val = self.syms[node.name]
            if hasattr(node, 'index'):
                return val[self.visit(node.index)]
            return val

        def visit_BinOp(self, node):
            return BIN_OP_INTERPRETER_MAP[node.op.__class__](
                self.visit(node.left), self.visit(node.right))

        def visit_CmpOp(self, node):
            return CMP_OP_INTERPRETER_MAP[node.op.__class__](
                self.visit(node.left), self.visit(node.right))

        def visit_UnOp(self, node):
            return UN_OP_INTERPRETER_MAP[node.op.__class__](
                self.visit(node.e))

        def visit_Assign(self, node):
            new_val = self.visit(node.val)
            if hasattr(node.ref, 'index'):
                i = self.visit(node.ref.index)
                self.syms[node.ref.name][i] = new_val
            else:
                self.syms[node.ref.name] = new_val

        def visit_For(self, node):
            min_range = self.visit(node.min)
            max_range = self.visit(node.max)
            if type(min_range) != int or type(max_range) != int:
                raise NotImplementedError("Runtime: Range values must be integers, got"
                    " ({}, {})".format(min_range, max_range))
            for i in range(min_range, max_range):
                self.syms[node.var] = i
                result = self.visit(node.body)
                if isinstance(result, Return):
                    return result

        def visit_If(self, node):
            cond_val = self.visit(node.cond)
            if cond_val:
                return self.visit(node.body)
            elif hasattr(node, 'elseBody'): # and not cond_val
                return self.visit(node.elseBody)

        def visit_Block(self, node):
            for stmt in node.body:
                assert isinstance(stmt, ast.AST)
                result = evaluator.visit(stmt)
                if isinstance(result, Return):
                    return result

        def visit_Return(self, node):
            return node

        def generic_visit(self, node):
            raise NotImplementedError("TODO: implement {}".format(node.__class__.__name__))
    
    evaluator = EvalExpr(syms)
    
    for stmt in ir.body.body:
        assert isinstance(stmt, ast.AST)
        result = evaluator.visit(stmt)
        if isinstance(result, Return):
            return evaluator.visit(result.val)

def Compile(f):
    """'Compile' the function f"""
    # Parse and extract the function definition AST
    fun = ast.parse(inspect.getsource(f)).body[0]
    print("Python AST:\n{}\n".format(astor.dump(fun)))
    
    simpleFun = PythonToSimple().visit(fun)
    
    print("Simple IR:\n{}\n".format(astor.dump(simpleFun)))
    
    # package up our generated simple IR in a 
    def run(*args):
        return Interpret(simpleFun, *args)
    
    return run


#############
## TEST IT ##
#############

# Define a trivial test program to start
def trivial() -> int:
    return 5

def math(x : int, y : int) -> int:
    return x + y - 4 * 3 / 2 % 4

def logic(x : int, y: int, z: int) -> int:
    return x and y or z

def fib(n : int) -> int:
    x = 1
    y = 1
    for i in range(n-2):
        tmp = x + y
        x = y
        y = tmp
        i += 4
    return y

def maxFour(x : int) -> int:
    if x > 4:
        return 4
    return x

def multiCmp(x : int, y : int, z : int) -> int:
    if x > y < z:
        return 1
    else:
        return 0

def absVal(x : int) -> int:
    if not x > 0:
        x = -x
    return x

def cmpOp() -> int:
    if 1 == 1 > 0 < 5 >= 5 <= 5 != 4:
        return 1
    return 0

# Confirm variable scoping works as expected.
def slowIdentity(x : int) -> int:
    for i in range(x):
        x = x
    return i

def subscript(arr, i : int) -> int:
    return arr[i]

def doubleIfTruthy(x : int, cond : int) -> int:
    x *= x and cond and 2
    return x

def test_it():
    trivialInterpreted = Compile(trivial)
    # run the original and our version, checking that their output matches:
    assert trivial() == trivialInterpreted()
   
    mathInterpreted = Compile(math) 
    assert math(4,3) == mathInterpreted(4,3)

    logicInterpreted = Compile(logic)
    # Confirm the `2` result to make sure the test case is WAI :)
    assert logic(1, 0, 2) == logicInterpreted(1, 0, 2) == 2

    fibInterpreted = Compile(fib)
    for i in range(1, 10):
        assert fib(i) == fibInterpreted(i)

    maxFourInterpreted = Compile(maxFour)
    assert maxFour(3) == maxFourInterpreted(3)
    assert maxFour(6) == maxFourInterpreted(6)

    multiCmpInterpreted = Compile(multiCmp)
    for x,y,z in itertools.permutations([1,2,3]):
        assert multiCmp(x, y, z) == multiCmpInterpreted(x, y, z)

    absValInterpreted = Compile(absVal)
    assert absVal(3.5) == absValInterpreted(3.5)
    assert absVal(-3.5) == absValInterpreted(-3.5)

    cmpOpInterpreted = Compile(cmpOp)
    assert cmpOp() == cmpOpInterpreted() == 1

    slowIdentityInterpreted = Compile(slowIdentity)
    assert slowIdentity(10) == slowIdentityInterpreted(10)

    subscriptInterpreted = Compile(subscript)
    assert subscript([1, 2, 3], 1) == subscriptInterpreted([1, 2, 3], 1)

    doubleIfTruthyInterpreted = Compile(doubleIfTruthy)
    assert doubleIfTruthy(5, 1) == doubleIfTruthyInterpreted(5, 1) == 10
    assert doubleIfTruthy(5, 0) == doubleIfTruthyInterpreted(5, 0) == 0
    assert doubleIfTruthy(0, 2) == doubleIfTruthyInterpreted(0, 2) == 0

if __name__ == '__main__':
    test_it()

# # Version of the interpreter that modifies the list of statements to be 
# # operated on instead of visiting them in a tree-like manner.
# def Interpret(ir, *args):
#     assert isinstance(ir, FuncDef)
    
#     # Initialize a symbol table, to store variable => value bindings
#     syms = dict(zip(ir.args, args))

    
#     # Build a visitor to evaluate Exprs, using the symbol table to look up
#     # variable definitions
#     class EvalExpr(ast.NodeVisitor):
#         def __init__(self, symbolTable):
#             self.syms = symbolTable
        
#         def visit_IntConst(self, node):
#             return node.val

#         def visit_Ref(self, node):
#             val = self.syms[node.name]
#             if hasattr(node, 'index'):
#                 return val[self.visit(node.index)]
#             return val

#         def visit_BinOp(self, node):
#             return BIN_OP_INTERPRETER_MAP[node.op.__class__](
#                 self.visit(node.left), self.visit(node.right))

#         def visit_CmpOp(self, node):
#             return CMP_OP_INTERPRETER_MAP[node.op.__class__](
#                 self.visit(node.left), self.visit(node.right))

#         def visit_UnOp(self, node):
#             return UN_OP_INTERPRETER_MAP[node.op.__class__](
#                 self.visit(node.e))

#         def visit_Assign(self, node):
#             new_val = self.visit(node.val)
#             if hasattr(node.ref, 'index'):
#                 i = self.visit(node.ref.index)
#                 self.syms[node.ref.name][i] = new_val
#             else:
#                 self.syms[node.ref.name] = new_val

#         def visit_For(self, node):
#             min_range = self.visit(node.min)
#             max_range = self.visit(node.max)
#             if type(min_range) != int or type(max_range) != int:
#                 raise NotImplementedError("Runtime: Range values must be integers, got"
#                     " ({}, {})".format(min_range, max_range))
#             new_stmts = []
#             for i in range(min_range, max_range):
#                 new_stmts.append(Assign(ref=Ref(name=node.var), val=IntConst(val=i)))
#                 new_stmts.extend(self.visit(node.body))
#             return new_stmts

#         def visit_If(self, node):
#             cond_val = self.visit(node.cond)
#             if cond_val:
#                 return self.visit(node.body)
#             elif hasattr(node, 'elseBody'): # and not cond_val
#                 return self.visit(node.elseBody)

#         def visit_Block(self, node):
#             return node.body

#         def visit_Return(self, node):
#             return self.visit(node.val)

#         def generic_visit(self, node):
#             raise NotImplementedError("TODO: implement {}".format(node.__class__.__name__))
    
#     evaluator = EvalExpr(syms)
   
#     stmts = ir.body.body[:]
#     while len(stmts) > 0:
#         stmt = stmts.pop(0)
#         assert isinstance(stmt, ast.AST)
#         if isinstance(stmt, Return):
#             return evaluator.visit(stmt)
#         result = evaluator.visit(stmt)
#         if type(result) == list and len(result) > 0 and isinstance(result[0], ast.AST):
#             stmts = result + stmts
