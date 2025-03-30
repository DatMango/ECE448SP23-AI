'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    #raise RuntimeError("You need to write this part!")
    standardized_rules = dict()
    variables = []
    total_variables = 0

    for ids, rules in nonstandard_rules.items():
        if "something" not in rules['consequent']:
            something_nonexistent = True

            for antecedent in rules['antecedents']:
                if "something" in antecedent:
                  something_nonexistent = False

            if something_nonexistent:
                standardized_rules[ids] = copy.deepcopy(rules)
                continue
      
        total_variables += 1
        standard_rule = dict()
        standard_rule['text'] = rules['text']
        standard_antecedents = copy.deepcopy(rules['antecedents'])
        standard_consequent = rules['consequent'][:]

        for i in range(0, len(standard_antecedents)):
            for j in range(0, len(standard_antecedents[i])):
                if standard_antecedents[i][j] == "something":
                    standard_antecedents[i][j] = "var" + str(total_variables)
        for i in range(0, len(standard_consequent)):
            if standard_consequent[i] == "something":
                standard_consequent[i] = "var" + str(total_variables)
        
        variables.append("var" + str(total_variables))
        standard_rule['antecedents'] = standard_antecedents
        standard_rule['consequent'] = standard_consequent
        standardized_rules[ids] = standard_rule

    return standardized_rules, variables

def search(x, sub):
    if x not in sub:
        return x
    else:
      return search(sub[x], sub)

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    #raise RuntimeError("You need to write this part!")
    if query[-1] == True and datum[-1] == False or len(query) != len(datum):
        return None, None
    
    unification = []
    subs = dict()

    for i in range(0, len(query)):
        if query[i] in variables and datum[i] in variables:
            idx = search(query[i], subs)
            subs[idx] = datum[i]
        elif query[i] in variables:
            idx = search(query[i], subs)
            subs[idx] = datum[i]
        elif datum[i] in variables:
            idx = search(datum[i], subs)
            subs[idx] = query[i]
        elif query[i] != datum[i]:
            return None, None
    
    for val in query:
        unification.append(search(val, subs))

    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    #raise RuntimeError("You need to write this part!")
    applications = []
    goalsets = []

    for goal in goals:
        temp_goal = list(goals[:])
        temp_app = dict()
        temp_app['antecedents'] = []
        unifi, subs = unify(goal, rule['consequent'], variables)
        
        if subs is None:
            continue
        
        temp_goal.remove(goal)
        for antece in rule['antecedents']:
            temp_antece = antece[:]
            for i in range(0, len(temp_antece)):
                temp_antece[i] = search(temp_antece[i], subs)
            temp_goal.append(tuple(temp_antece))
            temp_app['antecedents'].append(temp_antece)
            
        temp_conseq = rule['consequent'][:]

        for i in range(0, len(temp_conseq)):
            temp_conseq[i] = search(temp_conseq[i], subs)
        temp_app['consequent'] = temp_conseq
        applications.append(temp_app)
        goalsets.append(tuple(temp_goal))

    return applications, goalsets

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    #raise RuntimeError("You need to write this part!")
    proof = []

    queue = []
    applic = dict()
    prev_pts = dict()
    queue.append((tuple(query),))
    prev_pts[(tuple(query),)] = None

    while queue:
        curr = queue.pop(0)
        for rule in rules.values():
            applics, gss = apply(rule, curr, variables)
            temp_list = zip(gss, applics)
            for temp_goal, temp_app in temp_list:
                temp_app['text'] = rule['text']

                if not temp_goal:
                    prev_pts["end"] = tuple(curr)
                    applic["end"] = temp_app
                    break
                
                queue.append(tuple(temp_goal))
                prev_pts[tuple(temp_goal)] = tuple(curr)
                applic[tuple(temp_goal)] = temp_app
      
    if "end" not in prev_pts:
        return None
    
    temp = "end"
    while temp != (tuple(query),):
        proof.append(applic[temp])
        temp = prev_pts[temp]
    
    return proof


