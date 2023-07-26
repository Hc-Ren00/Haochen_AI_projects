Name: Haochen Ren, Tengzhi Zhuo

Description: my evaluation() function is different from the original score() function since it also focuses on dealing with open nodes nearby a sequence of same numbers. In this way, the bot knows that there is a chance to have 3-in-a-rows or more which will gain more points.

Test Description: 'normal case' tests for correctness of minimax. 'more choices' also checks the correctness of minimax but offered the bot for more choices. 'test_evaluation' is used to test my heuristic function where scores of two players are almost equal. 'test_evaluation2' tests the heuristic function works fine.

This AI player that I created won the third place in the competition with other 100+ students, with win/loss/draw as 221/20/9.