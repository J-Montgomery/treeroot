from dataclasses import dataclass
from typing import Any
import numpy as np

# --- AST Base ---
class Matcher:
    # Overload bitwise operators to build expression trees
    def __and__(self, other): return And(self, other)
    def __or__(self, other):  return Or(self, other)
    def __invert__(self):     return Not(self)

@dataclass(frozen=True)
class Always(Matcher):
    pass

@dataclass(frozen=True)
class Never(Matcher):
    pass

@dataclass(frozen=True)
class Not(Matcher):
    m: Matcher

@dataclass(frozen=True)
class And(Matcher):
    lhs: Matcher
    rhs: Matcher

@dataclass(frozen=True)
class Or(Matcher):
    lhs: Matcher
    rhs: Matcher

# --- Atomic Field Matchers ---
@dataclass(frozen=True)
class FieldEq(Matcher):
    field: str
    value: Any

@dataclass(frozen=True)
class FieldNe(Matcher):
    field: str
    value: Any

@dataclass(frozen=True)
class FieldLt(Matcher):
    field: str
    value: float

@dataclass(frozen=True)
class FieldGe(Matcher):
    field: str
    value: float

@dataclass(frozen=True)
class FieldIn(Matcher):
    field: str
    bitmap: np.ndarray

# --- Customization Point: Negate ---
def negate(m: Matcher) -> Matcher:
    if isinstance(m, Always):  return Never()
    if isinstance(m, Never):   return Always()
    if isinstance(m, Not):     return m.m # unwrap
    if isinstance(m, FieldEq): return FieldNe(m.field, m.value)
    if isinstance(m, FieldNe): return FieldEq(m.field, m.value)
    if isinstance(m, FieldLt): return FieldGe(m.field, m.value)
    if isinstance(m, FieldGe): return FieldLt(m.field, m.value)

    return Not(m)

# --- Customization Point: Implication (A => B) ---
def implies(a: Matcher, b: Matcher) -> bool:
    if a == b: return True # X => X is true
    if isinstance(a, Never) or isinstance(b, Always): return True # false => X, X => true
    
    # Compound Logic Implication
    if isinstance(a, And): return implies(a.lhs, b) or implies(a.rhs, b)
    if isinstance(b, Or):  return implies(a, b.lhs) or implies(a, b.rhs)
    
    # Domain-Specific Relational Warrants
    if isinstance(a, FieldEq) and isinstance(b, FieldEq):
        if a.field == b.field: return a.value == b.value
    
    if isinstance(a, FieldEq) and isinstance(b, FieldIn):
        if a.field == b.field: return a.value in b.values
        
    if isinstance(a, FieldLt) and isinstance(b, FieldLt):
        if a.field == b.field: return a.value <= b.value
        
    if isinstance(a, FieldEq) and isinstance(b, FieldLt):
        if a.field == b.field: return a.value < b.value

    if isinstance(a, FieldGe) and isinstance(b, FieldLt):
        if a.field == b.field and a.value >= b.value: return True

    if isinstance(a, FieldGe) and isinstance(b, FieldGe):
        if a.field == b.field: return a.value >= b.value

    return False

# --- Simplification Engine ---
def simplify(m: Matcher) -> Matcher:
    if isinstance(m, Not):
        return negate(simplify(m.m))

    if isinstance(m, And):
        l, r = simplify(m.lhs), simplify(m.rhs)
        if implies(l, r): return l  
        if implies(r, l): return r  
        # Contradiction check: implies(l, negate(r)) -> never
        if implies(l, negate(r)) or implies(r, negate(l)): return Never()
        return And(l, r)

    if isinstance(m, Or):
        l, r = simplify(m.lhs), simplify(m.rhs)
        if implies(l, r): return r  
        if implies(r, l): return l  
        # Tautology check: implies(negate(l), r) -> always
        if implies(negate(l), r) or implies(negate(r), l): return Always()
        return Or(l, r)

    return m

# --- Disjunctive Normal Form (Sum of Products) ---
def sum_of_products(m: Matcher) -> Matcher:
    m = simplify(m)
    if isinstance(m, Not):
        return m 
        
    if isinstance(m, Or):
        return Or(sum_of_products(m.lhs), sum_of_products(m.rhs))
        
    if isinstance(m, And):
        l = sum_of_products(m.lhs)
        r = sum_of_products(m.rhs)
        # Distributive Law
        if isinstance(l, Or):
            return Or(sum_of_products(And(l.lhs, r)), sum_of_products(And(l.rhs, r)))
        if isinstance(r, Or):
            return Or(sum_of_products(And(l, r.lhs)), sum_of_products(And(l, r.rhs)))
        return And(l, r)
        
    return m