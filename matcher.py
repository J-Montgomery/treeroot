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

@dataclass(frozen=True)
class And(Matcher):
    lhs: Matcher
    rhs: Matcher

    @staticmethod
    def from_list(matchers: list[Matcher]) -> Matcher:
        if not matchers: return Always()
        if len(matchers) == 1: return matchers[0]
        mid = len(matchers) // 2
        return And(And.from_list(matchers[:mid]), And.from_list(matchers[mid:]))

@dataclass(frozen=True)
class Or(Matcher):
    lhs: Matcher
    rhs: Matcher

    @staticmethod
    def from_list(matchers: list[Matcher]) -> Matcher:
        if not matchers: return Never()
        if len(matchers) == 1: return matchers[0]
        mid = len(matchers) // 2
        return Or(Or.from_list(matchers[:mid]), Or.from_list(matchers[mid:]))

# --- Customization Point: Negate ---
def negate(m: Matcher) -> Matcher:
    # Terminals
    if isinstance(m, Always):  return Never()
    if isinstance(m, Never):   return Always()
    
    # Double Negation: !!A -> A
    if isinstance(m, Not):     return m.m 
    
    # Leaf Swaps (Crucial for SIMD efficiency)
    if isinstance(m, FieldEq): return FieldNe(m.field, m.value)
    if isinstance(m, FieldNe): return FieldEq(m.field, m.value)
    if isinstance(m, FieldLt): return FieldGe(m.field, m.value)
    if isinstance(m, FieldGe): return FieldLt(m.field, m.value)

    # For logical branches (And/Or), we just wrap them.
    # This keeps the AST simple and avoids DNF explosion.
    return Not(m)

# --- Customization Point: Implication (A => B) ---
def implies(a: Matcher, b: Matcher) -> bool:
    if a is b: return True 
    
    # 1. Structural Logic (Must remain recursive)
    if isinstance(a, Never) or isinstance(b, Always): return True
    if isinstance(a, And): return implies(a.lhs, b) or implies(a.rhs, b)
    if isinstance(b, Or):  return implies(a, b.lhs) or implies(a, b.rhs)
    
    # 2. Fast Type & Field Guard
    type_a = type(a)
    type_b = type(b)
    
    # Using a tuple for 'in' is faster for small collections
    if type_a not in (FieldEq, FieldLt, FieldGe, FieldNe, FieldIn): return False
    if type_b not in (FieldEq, FieldLt, FieldGe, FieldNe, FieldIn): return False
    if a.field != b.field: return False
            
    # 3. Flattened Dispatch Table
    # We use local variable caching to avoid repeated attribute lookups
    v_a = a.value if type_a is not FieldIn else None
    v_b = b.value if type_b is not FieldIn else None

    # --- Type A: FieldEq ---
    if type_a is FieldEq:
        if type_b is FieldEq: return v_a == v_b
        if type_b is FieldLt: return v_a <  v_b
        if type_b is FieldGe: return v_a >= v_b
        if type_b is FieldNe: return v_a != v_b
        if type_b is FieldIn:
            # Inline bitmap check
            idx = v_a >> 3
            return idx < len(b.bitmap) and (b.bitmap[idx] & (1 << (v_a & 7))) != 0

    # --- Type A: FieldLt ---
    if type_a is FieldLt:
        if type_b is FieldLt: return v_a <= v_b
        if type_b is FieldNe: return v_a <= v_b

    # --- Type A: FieldGe ---
    if type_a is FieldGe:
        if type_b is FieldGe: return v_a >= v_b
        if type_b is FieldNe: return v_a > v_b

    # --- Type A: FieldNe ---
    if type_a is FieldNe:
        if type_b is FieldNe: return v_a == v_b

    if type_a is FieldLt and type_b is FieldGe and v_a <= v_b:
        # (x < 10) AND (x >= 30) is impossible
        return False # a doesn't imply b, but they're disjoint

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