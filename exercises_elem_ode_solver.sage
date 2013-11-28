"""
This module contains various functions to solve first order ordinary
differential equations. For examples, see the docstring for the solve_ode
method.

To run the file use the following command in the SAGE shell.

sage: runfile 'exercises_elem_ode_solver.sage'

To run the file as a module, you need to preparse the module as a python file;
i.e., use the command in a linux shell (NOT in the SAGE shell):

$ sage --preparse exercises_elem_ode_solver.sage

This will create a file exercises_elem_ode_solver.py which can then be imported
as a module like in normal python. Then you are free to run the solve_ode
method.

Note that I have not included proper SAGE docstring formatting in this 
module-docstring because I have not found a way to actually see this docstring
in the SAGE command shell.

Author: John Kluesner
Date:   28 Nov, 2013
Email:  stringman45@gmail.com
"""

def solve_ode(w, x1, y1):
    r"""
    Attempt to find a symbolic solution to a first-order differential equation.

    INPUT:

    - ``w`` -- An expression representing a differential equation that can be
      transformed to the form M + N*D[0](y)(x) = 0.
    - ``x1`` -- The independent variable of ``w``.
    - ``y1`` -- the dependent variable of ``w``.

    OUTPUT:

    - An implicit solution to the differential equation or False.

    EXAMPLES:

    This example illustrates a solution by seperation of variables::

        sage: x = var('x')
        sage: y = function('y', x)
        sage: diff_eq = y.diff(x) == 2*x*y^2
        sage: solve_ode(diff_eq, x, y)
        -x^2 - 1/y(x) == c

    This example illustrates a solution by an integrating factor::

        sage: x = var('x')
        sage: y = function('y', x)
        sage: diff_eq = (3 + 3*y^2/x)*y.diff(x) == -2*x - 3*y/x
        sage: solve_ode(diff_eq, x, y)
        2/3*x^3 + y(x)^3 + 3*x*y(x) == c

    This example illustrates a solution with a differential equation that can
    be transformed into function of (y/x)::

        sage: x = var('x')
        sage: y = function('y', x)
        sage: diff_eq = (x - y)*y.diff(x) == x + y
        sage: solve_ode(diff_eq, x, y)
        -1/2*log(y(x)^2/x^2 + 1) - log(x) + arctan(y(x)/x) == c

    This example illustrates a solution that can't be found::

        sage: x = var('x')
        sage: y = function('y', x)
        sage: diff_eq = y.diff(x) == (x + y + 4)/(x - y - 6)
        sage: solve_ode(diff_eq, x, y)
        False

    .. NOTE::

        All of the methods in this module are based on exercises in Joel S.
        Cohen's book Computer Algebra And Symbolic Computation: Elementary
        Algorithms. The three methods used by solve_ode are:

        (1) Seperation of variables.

        (2) Integrating Factor.

        (3) Check if homogeneous and solve using seperation of variables.

    .. WARNING::

        This method potentially adds new variables to the variable namespace.
        In particular, ``c`` is added as it is used as the constant of
        integration and ``z`` is added as it is used to check if the solution
        can be found by a particular substitution. This could overwrite an
        object ``c`` or ``z`` you are already using. I have not found a way to
        avoid this. A simple idea would be to use _z because that is likely not
        in use. Also, if you use ``y`` as your dependent variable it recasts it
        as a variable object instead of a function of ``x``, even though the
        output doesn't suggest this. You will have to redefine ``y``. A good
        solution to this problem would be to use ``_y`` instead as well.
        Because this code is mostly used as an educational exercise, I have not
        bothered to make these changes.

    AUTHORS:

    -John Kluesner (2013-05-27)
    """
    coeffs = transform_ode(w, x1, y1)
    x = var('%r'%(x1))
    y = var(('%r'%(y1))[0 : len(y1.__str__ ())-len(x1.__str__())-2])
    M = coeffs[0].subs({y1:y})
    N = coeffs[1].subs({y1:y})
    solution = seperable_ode(M, N, x, y)
    if solution is False:
        solution = solve_exact(M, N, x, y)
    if solution is False:
        solution = homogeneous(M, N, x, y)
    if solution is False:
        return solution
    else:
        return solution.subs({x:x1, y:y1})

def seperate_variables(u, x, y):
    r"""
    If an algebraic expression ``u`` is seperable in ``x`` and ``y``, (i.e., it
    can be solved by the method of seperation of variables in a differential
    equation, dy/dx = ``u``), then return a list of the x-terms and y-terms. 
    Otherwise, return False.

    INPUT:

    - ``u`` -- An algebraic expression of 2 variables.
    - ``x`` -- The independent variable of ``u``.
    - ``y`` -- the dependent variable of ``u``. 
    
    OUTPUT:

    - A list of the x-terms and the y-terms in a seperable algebraic
      expression or False if not seperable.

    EXAMPLES:

    The following examples illustrate the basic functionality.::

        sage: x,y = var('x,y')
        sage: seperate_variables(3*x^2*y + 3*x*y, x, y)
        [3*(x + 1)*x, y]
        sage: seperate_variables(x + y, x, y)
        False
        sage: u = exp((x + y)*(x - y))
        sage: seperate_variables(u, x, y)
        [e^(x^2), e^(-y^2)]
        sage: seperate_variables(x^2*y^3 + x^2*y^2 + x*y^3 + x*y^2, x, y)
        [(x + 1)*x, (y + 1)*y^2]
        sage: seperate_variables(2^((x+y)*(x+y)), x, y)
        False

    However, some expressions that should work aren't checked for.::

        sage: seperate_variables(1/2*(sin(x + y) + sin(x - y)), x, y)
        False
        sage: seperate_variables(log(x^y), x, y)
        False

    We leave this in the hands of the user.::

        sage: u = 1/2*(sin(x + y) + sin(x - y))
        sage: seperate_variables(u.simplify_trig(), x, y)
        [sin(x), cos(y)]
        sage: seperate_variables(log(x^y).expand_log(), x, y)
        [log(x), y]

    AUTHORS:

    -John Kluesner (2013-05-25)
    """
    x_terms = 1
    y_terms = 1
    if not type(u) is Expression:
        x_terms = u
    elif not u.has(x):
        y_terms = u
    elif not u.has(y):
        x_terms = u
    elif u.operator() is operator.mul:
        for op in u.operands():
            if not op.has(y):
                x_terms = x_terms*op
            elif not op.has(x):
                y_terms = y_terms*op
            else:
                new_terms = seperate_variables(op, x, y)
                if new_terms is False:
                    return False
                x_terms = x_terms*new_terms[0]
                y_terms = y_terms*new_terms[1]
    elif u.operator() is operator.add:
        v = u.factor()
        if v.operator() is operator.add:
            return False
        else:
            new_terms = seperate_variables(v, x, y)
            if new_terms is False:
                return False
            x_terms = x_terms*new_terms[0]
            y_terms = y_terms*new_terms[1]
    elif u.operator() is operator.pow:
        v = u.expand()
        if v.operator() is operator.pow:
            return False
        else:
            new_terms = seperate_variables(v, x, y)
            if new_terms is False:
                return False
            x_terms = x_terms*new_terms[0]
            y_terms = y_terms*new_terms[1]
    elif u.operator() is exp:
        v = u.operands()[0]
        if v.operator() in [operator.mul, operator.pow]:
            v = v.expand()
            if v.operator() in [operator.mul, operator.pow]:
                return False
        if v.operator() is operator.add:
            for op in v.operands():
                if not op.has(y):
                    x_terms = x_terms*exp(op)
                elif not op.has(x):
                    y_terms = y_terms*exp(op)
                else:
                    return False
    else:
        return False
    return [x_terms, y_terms]

def transform_ode(w, x, y):
    r"""
    Transform a general first order differential equation into a more usable
    form, M + N*D[0](y)(x) == 0, and return M and N in a list.

    INPUT:

    - ``w`` -- An expression representing a differential equation that can be
      transformed to the form M + N*D[0](y)(x) == 0.
    - ``x`` -- The independent variable of ``w``.
    - ``y`` -- the dependent variable of ``w``. 
    
    OUTPUT:

    - The list [M, N] where M is the coefficient of D[0](y)(x) of degree 0 and
      N is the coefficent of D[0](y)(x) of degree 1.

    EXAMPLES:

        See solve_ode.

    .. Notes::

        This method is used by seperable_ode. It is probably not useful by
        itself.

    AUTHORS:

    -John Kluesner (2013-05-27)
    """
    v = (w.operands()[0] - w.operands()[1]).simplify_rational()
    n = v.numerator()
    M = n.coeff(y.diff(x), 0)
    N = n.coeff(y.diff(x), 1)
    return [M,N]

def solve_exact(M, N, x, y):
    r"""
    Return the implicit solution to a first order differential equation using
    the integrating factor process, or False if failure.

    INPUT:

    - ``M`` -- An algebraic expression representing the coefficient of
      D[0](y)(x) of degree 0.
    - ``N`` -- An algebraic expression representing the coefficient of
      D[0](y)(x) of degree 1.
    - ``x`` -- The independent variable of ``M`` and ``N``.
    - ``y`` -- the dependent variable of ``M`` and ``N``. 
    
    OUTPUT:

    - Implicit solution to a first order differential equation using
      the integrating factor process, or False if failure.

    EXAMPLES:

        See solve_ode.

    .. Notes::

        This method is used by solve_ode. It is probably not useful by itself.

    AUTHORS:

    -John Kluesner (2013-05-27)
    """
    c = var('c')
    if bool(N == 0):
        return False
    elif bool(M == 0):
        return y == c
    M_diff_y = M.diff(y)
    N_diff_x = N.diff(x)
    d = M_diff_y - N_diff_x
    if bool(d == 0):
        integrating_factor = 1
    else:
        F = (d/N).rational_simplify()
        if not F.has(y):
            integrating_factor = exp(F.integrate(x))
            d = 0
        else:
            G = (-d/M).rational_simplify()
            if not G.has(x):
                integrating_factor = exp(G.integrate(y))
                d = 0
            else:
                z = var('z')
                R = (d/(N*y - M*x)).rational_simplify()
                S = R.subs({x:z/y}).rational_simplify()
                if not S.has(y):
                    integrating_factor = (exp(R.integrate(z))).subs({z:x*y})
                    d = 0
    if bool(d == 0):
        solution = integrating_factor*M.integrate(x)
        int_const_deriv = integrating_factor*N - solution.diff(y)
        int_const = int_const_deriv.integrate(y)
        return solution + int_const == c
    else:
        return False 

def seperable_ode(M, N, x, y):
    r"""
    Return the implicit solution to a first order differential equation using
    the speration of variables process, or False if failure.

    INPUT:

    - ``M`` -- An algebraic expression representing the coefficient of the
      D[0](y)(x) term of degree 0.
    - ``N`` -- An algebraic expression representing the coefficient of the
      D[0](y)(x) term of degree 1.
    - ``x`` -- The independent variable of ``M`` and ``N``.
    - ``y`` -- the dependent variable of ``M`` and ``N``. 
    
    OUTPUT:

    - Implicit solution to a first order differential equation using
      the seperation of variables process, or False if failure.

    EXAMPLES:

        See solve_ode.

    .. Notes::

        This method is used by solve_ode. It is probably not useful by itself.

    AUTHORS:

    -John Kluesner (2013-05-27)
    """
    seperated_list = seperate_variables(-M/N, x, y)
    c = var('c')
    if seperated_list is False:
        return False
    else:
        x_terms, y_terms = seperated_list
        return (1/y_terms).integrate(y) - (x_terms.integrate(x)) == c

def homogeneous(M, N, x, y):
    r"""
    Return the implicit solution to a first order differential equation using
    the homogeneous solution technique (z = y/x), or False if failure.

    INPUT:

    - ``M`` -- An algebraic expression representing the coefficient of the
      D[0](y)(x) term of degree 0.
    - ``N`` -- An algebraic expression representing the coefficient of the
      D[0](y)(x) term of degree 1.
    - ``x`` -- The independent variable of ``M`` and ``N``.
    - ``y`` -- the dependent variable of ``M`` and ``N``. 
    
    OUTPUT:

    - Implicit solution to a first order differential equation using
      the homogeneous solution technique (z = y/x), or False if failure.

    EXAMPLES:

        See solve_ode.

    .. Notes::

        This method is used by solve_ode. It is probably not useful by itself.

    AUTHORS:

    -John Kluesner (2013-05-27)
    """
    c = var('c')
    z = var('z')
    R = (-M/N).subs({y: x*z}).rational_simplify()
    if not R.has(x):
        solution = (1/(R - z)).integrate(z) - log (x) == c
        solution = solution.subs({z:y/x})
        return solution
    else:
        return False
