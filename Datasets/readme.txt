Datasets

Carrera Track
A Carrera track is a platform for slot car physical simulation where the cars in a lane
are accelerated by adjusting the voltage given to that car. Generally, there are
controllers with the car for controlling the voltage. But for the experimental setup,
these voltages were regulated by a Raspberry Pi computer connected to the main
computer.
The cars on the track are in a platoon formation meaning three cars in a row with a
fixed distance from the lead car. The formation is done by mounting a camera on top
of the Carrera track, which uses Cooperative Adaptive Cruise Control and sends the
commands through Raspberry Pi to maintain the platoon formation. The camera is
also used to record our position, speed, acceleration and header data fields.

2. Veremi Dataset with Extension
VeReMi dataset came at the end of 2018, and the main idea behind the paper was to
serve as a reference dataset that can be used for misbehaviour frameworks. An
extension of this dataset came out in the mid of 2020, which included more attacks
and experiments with varying vehicle densities. Each attack consists of log files from
each car, which contains the messages they received, and a ground truth file
containing the true messages. A single message entry contains sender ids, position,
speed, acceleration and header distance fields. To evaluate the state of the art
results against our models, we used only the best and worst attacks data.

● ConstPos. This causes the attacker cars to transmit the same location
regardless of where they are currently located in the simulation.

● DataReplaySybil. This causes the cars to replay the same messages they
have already received from other vehicles. The messages that they send can
come from any of the other cars that they have received messages from.

