class census:
 
    params = 13

    input_bounds = []
    input_bounds.append([1, 9]) # age
    input_bounds.append([0, 7]) # workclass
    input_bounds.append([0, 39]) #69 for THEMIS  fnlwgt
    input_bounds.append([0, 15]) # education
    input_bounds.append([0, 6]) # marital_status
    input_bounds.append([0, 13]) # occupation
    input_bounds.append([0, 5]) # relationship
    input_bounds.append([0, 4]) # race
    input_bounds.append([0, 1]) #  sex
    input_bounds.append([0, 99]) # capital_gain
    input_bounds.append([0, 39]) # capital_loss
    input_bounds.append([0, 99]) # hours_per_week
    input_bounds.append([0, 40]) # native_country
    protected_params = [0, 7, 8]
    input_bounds_size=[]
    for x in input_bounds:
        input_bounds_size.append(x[1]-x[0])
    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    all_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    protected_params = [8, 12]
    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
                    "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                     "residence", "property_magnitude", "age", "other_payment_plans", "housing",
                    "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]
    
    all_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])
    protected_params = [0]
    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                    "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    all_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class meps:
    params = 40
    input_bounds = [
        [0, 3],
        [0, 85],
        [0, 1],
        [0, 1],
        [0, 9],
        [0, 3],
        [0, 3],
        [0, 3],
        [0, 5],
        [0, 5],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 3],
        [0, 1],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
        [-9, 70],
        [-9, 75],
        [-9, 24],
        [0, 7],
        [0, 4],
        [0, 4],
        [0, 2],
    ]
    # the name of each feature
    feature_name = [
        'REGION',
        'AGE',
        'SEX',
        'RACE',
        'MARRY',
        'FTSTU',
        'ACTDTY',
        'HONRDC',
        'RTHLTH',
        'MNHLTH',
        'CHDDX',
        'ANGIDX',
        'MIDX',
        'OHRTDX',
        'STRKDX',
        'EMPHDX',
        'CHBRON',
        'CHOLDX',
        'CANCERDX',
        'DIABDX',
        'JTPAIN',
        'ARTHDX',
        'ARTHTYPE',
        'ASTHDX',
        'ADHDADDX',
        'PREGNT',
        'WLKLIM',
        'ACTLIM',
        'SOCLIM',
        'COGLIM',
        'DFHEAR42',
        'DFSEE42',
        'ADSMOK42',
        'PCS42',
        'MCS42',
        'K6SUM42',
        'PHQ242',
        'EMPST',
        'POVCAT',
        'INSCOV',
    ]

    protected_params = [0, 1, 2]
    # the name of each class
    class_name = ["no", "yes"]
    
    all_param = []
    for i in range(params):
        all_param.append(i)

class ricci:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 5

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([40, 92])
    input_bounds.append([46, 95])
    input_bounds.append([0, 2])  # race
    input_bounds.append([45, 92])
    protected_params = [3]
    
    all_param = []
    for i in range(params):
        all_param.append(i)
    

class tae:
    params = 5
    input_bounds = []
    input_bounds.append([1, 2])   #whether_is_native_or_not
    input_bounds.append([1, 25])
    input_bounds.append([1, 26])
    input_bounds.append([1, 2])
    input_bounds.append([3, 66])
    protected_params = [0]
    
    all_param = []
    for i in range(params):
        all_param.append(i)

class student_math:
    params = 32
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([15, 22])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 3])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([0, 75])
    input_bounds.append([3, 19])
    input_bounds.append([0, 19])
    protected_params = [0]
    
    all_param = []
    for i in range(params):
        all_param.append(i)

class student_por:
    params = 32
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])        #sex
    input_bounds.append([15, 22])      #age
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 3])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([1, 5])
    input_bounds.append([0, 32])
    input_bounds.append([0, 19])
    input_bounds.append([0, 19])
    protected_params = [0]
    
    all_param = []
    for i in range(params):
        all_param.append(i)


class compas:
    params = 16
    input_bounds = []
    input_bounds.append([0, 1])
    input_bounds.append([18, 96])
    input_bounds.append([0, 2])
    input_bounds.append([0, 5])    #race
    input_bounds.append([0, 20])
    input_bounds.append([0, 13])
    input_bounds.append([0, 11])
    input_bounds.append([0, 43])
    input_bounds.append([-29, 52])
    input_bounds.append([0, 94])
    input_bounds.append([0, 12])
    input_bounds.append([0, 9])
    input_bounds.append([0, 2])
    input_bounds.append([0, 9])
    input_bounds.append([0, 2])
    input_bounds.append([0, 11])
    protected_params = [3]
    
    all_param = []
    for i in range(params):
        all_param.append(i)