import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
from helpers import proj, apply_move, size_band

start_date = '2019-08-31'

mydata = quandl.get("USTREASURY/YIELD", authtoken="s9UQbaVsDPbwxKr6HDVT", start_date=start_date)

# yields suggested by Hrusa
mydata = mydata[['2 YR', '3 YR', '5 YR', '7 YR' , '10 YR', '20 YR', '30 YR']]

#modifying the data
yields = np.array(mydata)
diffs = yields[1:, ] - yields[0:-1, ]
diffs = diffs.transpose()
diffs = np.nan_to_num(diffs, 0)

#eigendecomposition
covar = np.cov(diffs)
eigen = np.linalg.eig(covar)

eigenvalues = eigen[0]
eigenvectors = eigen[1]

varpct = eigenvalues / sum(eigenvalues) * 100
lastmove = diffs[:, -1]

pc1, pc2, pc3 = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]


#normalizing the "direction" of the components to something i find intuitively appealing.
#level: up
if pc1[0] < 0: pc1 = -1 * pc1

#make sure it's a steepener and not a flattener
if pc2[0] > 0: pc2 = -1 * pc2

#i don't really care what direction twist is but I intuitively prefer to think of it as rising short and long term
if pc3[0] < 0: pc3 = -1 * pc3
comp1, comp2, comp3 = proj(pc1, lastmove), proj(pc2, lastmove), proj(pc3, lastmove)

levelchanges = np.matmul(diffs.transpose(), pc1)
steepchanges = np.matmul(diffs.transpose(), pc2)
twistchanges = np.matmul(diffs.transpose(), pc3)


#calculate what amount of the last move was explained by each
levelchg = apply_move(diffs.transpose(), pc1)
slopechg = apply_move(diffs.transpose(), pc2)
twistchg = apply_move(diffs.transpose(), pc3)

v1 = levelchg[-1] * 100
v2 = slopechg[-1] * 100
v3 = twistchg[-1] * 100

belt_parameter = 0.5
l100, s100, t100 = levelchg * 100, slopechg * 100, twistchg * 100
potential_level = size_band(l100[0:-1], v1 - belt_parameter, v1 + belt_parameter, l100[1:])
potential_slope = size_band(s100[0:-1], v2 - belt_parameter, v2 + belt_parameter, s100[1:])
potential_twist = size_band(t100[0:-1], v3 - belt_parameter, v3 + belt_parameter, t100[1:])

if __name__ == '__main__':
    #plt.plot(mydata.tail().transpose())


    print('\n\nPrincipal components:')
    print(np.round(eigen[1], 2))


    print('\n\nVariance explained by each principal component:')
    print(np.round(varpct, 2))

    print('\n\nOne year ago:')
    print(mydata.head(1))

    print('\n\nMost recent levels:')
    print(mydata.tail())

    print('\n\n')
    print('Most recent change explained by level:', round(v1, 2), 'SD = ', round(np.std(levelchg) * 100, 2))
    print('Most recent change explained by slope:', round(v2, 2), 'SD = ', round(np.std(slopechg) * 100, 2))
    print('Most recent change explained by twist:', round(v3, 2), 'SD = ', round(np.std(twistchg) * 100, 2))


    levelchg = apply_move(diffs.transpose(), pc1)
    slopechg = apply_move(diffs.transpose(), pc2)
    twistchg = apply_move(diffs.transpose(), pc3)

    belt_parameter = 0.5
    l100, s100, t100 = levelchg * 100, slopechg * 100, twistchg * 100
    potential_level = size_band(l100[0:-1], v1 - belt_parameter, v1 + belt_parameter, l100[1:])
    potential_slope = size_band(s100[0:-1], v2 - belt_parameter, v2 + belt_parameter, s100[1:])
    potential_twist = size_band(t100[0:-1], v3 - belt_parameter, v3 + belt_parameter, t100[1:])

    print('\nExpected Level Change')
    print('Mean:', round(np.mean(potential_level), 2), 'sd:', round(np.std(potential_level), 2), 'n:', len(potential_level))

    print('\nExpected Slope Change')
    print('Mean:', round(np.mean(potential_slope), 2), 'sd:', round(np.std(potential_slope), 2), 'n:', len(potential_slope))

    print('\nExpected Twist Change')
    print('Mean:', round(np.mean(potential_twist), 2), 'sd:', round(np.std(potential_twist), 2), 'n:',
          len(potential_twist))


    plt.plot(np.cumsum(levelchg))
    plt.plot(np.cumsum(slopechg))
    plt.plot(np.cumsum(twistchg))

    plt.show()



