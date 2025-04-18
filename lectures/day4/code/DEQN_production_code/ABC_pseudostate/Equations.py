import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import alpha, beta, delta, tau, A_star, rho_ax, rho_dx, sigma_ax, sigma_dx


def equations(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    
    loss_dict = {}
    
    # policy today
    Ishare = PolicyState.Ishare_y(policy_state)
    Lamda = PolicyState.Lamda_y(policy_state)

    # output today
   # Y = State.a_x(state) * State.K_x(state) ** alpha 

    # investment today
    I = Ishare * Definitions.Y_x(state,policy_state)


    # capital tomorrow
    Knext = I + (1. - delta) * State.K_x(state)

    
    # consumption today
    C = tf.math.maximum(Definitions.Y_x(state,policy_state) - I, 1e-5) # max needed?

    # Fischer-Burmeister
    FB_a = 1 - (1. - delta) * State.K_x(state) / Knext
    FB_b = Lamda / C ** (-tau)
    

    
    # function which gives the RHS of the Euler equation for a given states tomorrow
    def RHS(snext, psnext):
        
        # policy tomorrow
        Isharenext = PolicyState.Ishare_y(psnext)
        Lamdanext = PolicyState.Lamda_y(psnext)
        
        # output tomorrow
        Ynext = State.a_x(snext) * State.K_x(snext) ** State.alpha_x(state) 

        # investment tomorrow
        Inext = Isharenext * Ynext
        
        # Return tomorrow
        Rnext = State.alpha_x(state) * State.a_x(snext) * State.K_x(snext) ** (State.alpha_x(state) - 1.) 
        
        # consumption tomorrow
        Cnext = tf.math.maximum(Ynext - Inext, 1e-5) # max needed?
        
        # RHS Euler Equation
        RHS =  State.d_x(snext) * (Cnext ** (-tau) * ( 1. - delta + Rnext) - Lamdanext * (1-delta))
        
        return RHS
    

    
    # Resource Constraint
    #tf.print("batch state: ", state)
    #loss_dict['eq_0'] = Definitions.Y_x(state,policy_state) - C - I
    loss_dict['eq_0'] = tf.math.multiply((Definitions.Y_x(state,policy_state) - C - I), (State.dummy_x(state)))
   
    
    # relative error EE
    
    #loss_dict['eq_1'] = Lamda - C ** (-tau) + State.beta_x(state) * E_t(lambda snext, psnext: RHS(snext, psnext))
    loss_dict['eq_1'] = tf.math.multiply((Lamda - C ** (-tau) + State.beta_x(state) * E_t(lambda snext, psnext: RHS(snext, psnext))), (State.dummy_x(state)))
    
    
    # error (Kuhn Tucker) Fischer Burmeister
    #loss_dict['eq_2'] = FB_a + FB_b - tf.sqrt(FB_a ** 2 + FB_b ** 2)
    loss_dict['eq_2'] = tf.math.multiply((FB_a + FB_b - tf.sqrt(FB_a ** 2 + FB_b ** 2)), (State.dummy_x(state)))
    
    





    return loss_dict
