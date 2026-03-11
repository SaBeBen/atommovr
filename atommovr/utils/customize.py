# File to customize plot colors, names of atoms, etc.

# Plot colors
SPECIES1COL = 'blue' # the color of the first atomic species (or all atoms, if single species)
SPECIES2COL = 'gold' # the color of the second atomic species (only for dual species)
NOATOMCOL = 'white' # would not recommend changing this from white, but you do you :)
EDGECOL = 'green' # the color of the circles around the atoms (to represent tweezers)
ARROWCOL = 'green' # the color of the arrows which indicate atom moves
EJECTCOL = 'green' # the color representing an atom that has been ejected
PICKUPFAILCOL = 'y' # the color of an atom that wasn't picked up by the tweezer and remained in the original spot
PUTDOWNFAILCOL = 'm' # the color of an atom that wasn't put down by the tweezer and was lost
COLLISIONFAILCOL = 'r' # the color of atoms that collided with each other and were lost
CROSSEDFAILCOL = 'r' # the color of atoms that were lost because of intersecting tweezer paths

# Atom names
SPECIES1NAME = 'Rb' # Rubidium
SPECIES2NAME = 'Cs' # Cesium

# Colorscheme from the paper
_nikhilgreen = [0.2,0.8,0.5]
_nikhilblue = [0.5,0.7,1]
_nikhilorange = [1,0.5,0.2]
_quantumviolet = '#53257F' # taken from Quantum journal
_quantumgray   = '#555555' # taken from Quantum journal