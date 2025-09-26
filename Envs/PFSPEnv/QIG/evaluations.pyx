### This function has been modified from the following licence ###

"""
MIT License

Copyright (c) 2018 js-aguiar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from numpy import zeros
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef taillard_acceleration(int[:] sequence, int[:,:] processing_times, int job_to_insert, int machines_number, int use_tie_breaking):
	
	cdef int ms[801]
	cdef int funct[801][61]
	cdef int position[801][61]
	cdef int approx[801][61]
	
	
	cdef int sequence_length, best_makespan, best_position
	cdef int i, j, ip, jp, pos
	cdef list best_makespans, best_positions
        
    
	sequence_length = len(sequence)
	ip = sequence_length+1
	
	for i in range(1,sequence_length+2):
		if i < sequence_length+1:
			position[i][0] = 0
			
			ip -= 1
			approx[ip][machines_number+1] = 0

		funct[i][0] = 0
		jp = machines_number+1
	
		for j in range(1, machines_number + 1):
			if i == 1:
				position[0][j] = 0
				approx[sequence_length+1][machines_number+1-j] = 0
			if i < sequence_length+1:
				
				jp -= 1

				if position[i][j-1] > position[i-1][j]:
					position[i][j] = position[i][j-1] + processing_times[sequence[i-1]-1,j-1]
				else:
					position[i][j] = position[i - 1][j] + processing_times[sequence[i-1]-1,j-1]

				if approx[ip][jp+1] > approx[ip + 1][jp]:
					approx[ip][jp] = approx[ip][jp + 1] + processing_times[sequence[ip-1]-1,jp-1]
				else:
					approx[ip][jp] = approx[ip + 1][jp] + processing_times[sequence[ip-1]-1,jp-1]

			if funct[i][j-1] > position[i-1][j]:
				funct[i][j] = funct[i][j-1] + processing_times[job_to_insert-1,j-1]
			else:
				funct[i][j] = position[i-1][j] + processing_times[job_to_insert-1,j-1]

    # makespam 
	best_makespan = 0
	best_position = 0
	best_makespans = []
	best_positions = []
    
	for i in range(1,sequence_length+2):
		ms[i] = 0
		for j in range(1,machines_number+1):
			pos = funct[i][j] +	approx[i][j]
			if pos > ms[i]:
				ms[i] = pos
	    # Check best insertion position
		best_makespans.append(ms[i])
		best_positions.append(i)
		if ms[i] < best_makespan or best_makespan == 0:
			best_makespan = ms[i]
			#best_makespans.append(best_makespan)
			best_position = i
			#best_positions.append(best_position)
	
	if use_tie_breaking > 0:
		best_position = tie_breaking(processing_times, position, funct, ms, job_to_insert, best_position, sequence_length, machines_number)
	return best_position, best_makespan, best_positions, best_makespans


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tie_breaking(int[:,:] processing_times, int[:,:] position, int[:,:] funct, int[:] ms,
				  int job_to_insert, int best_position, int sequence_length, int machines_number):
	
	cdef int best_makespan, num_ties, itbp
	cdef int itr, tie, i, j
	cdef int fl[801][61]

	best_makespan = ms[best_position]
	itbp = 100000000
	num_ties = 0

	for i in range(1, sequence_length + 2):
		if ms[i] == best_makespan:
			
			itr = 0
			num_ties += 1

			# If last position in sequence
			if i == sequence_length:
				for j in range(1, machines_number+1):
					itr += funct[sequence_length][j] - position[sequence_length-1][j] - processing_times[job_to_insert-1,j-1]

			# If not last position
			else:
				fl[i][1] = funct[i][1] + processing_times[i-1][0]
				for j in range(2, machines_number + 1):
					itr += funct[i][j] - position[i][j] + processing_times[i-1,j-1] - processing_times[job_to_insert-1,j-1]
					if fl[i][j-1] - funct[i][j] > 0:
						itr += fl[i][j-1] - funct[i][j]

					if fl[i][j-1] > funct[i][j]:
						fl[i][j] = fl[i][j-1] + processing_times[i-1,j-1]
					else:
						fl[i][j] = fl[i][j] + processing_times[i-1,j-1]

			if itr < itbp:
				best_position = i
				itbp = itr

	return best_position


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_completion_times(int[:] sequence, int[:,:] processing_times, int machines_number):
	
	cdef int sequence_length, i, j
	sequence_length = len(sequence)

	cdef int[:,::1] position = zeros((sequence_length+1,machines_number+1), dtype='int32')

	for i in range(1, sequence_length+1):
		position[i,0] = 0
		for j in range(1, machines_number+1):
			if i == 1:
				position[0,j] = 0

			if position[i-1,j] > position[i,j-1]:
				position[i,j] = position[i-1, j] + processing_times[sequence[i-1]-1,j-1]
			else:
				position[i,j] = position[i,j-1] + processing_times[sequence[i-1]-1,j-1]
	return position[sequence_length, machines_number]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef idle_times_calc(int[:] sequence, int[:,:] processing_times, int machines_number):
	
	cdef int sequence_length, i, j
	sequence_length = len(sequence)

	cdef int[:] idle_time = zeros((sequence_length), dtype='int32')
	cdef int[:,::1] position

	position = calculate_completion_times(sequence, processing_times, machines_number)

	for i in range(0, sequence_length):
		idle_time[sequence[i]-1] = 0
		for j in range(1,machines_number + 1):
			idle_time[sequence[i]-1] += position[i+1,j] - processing_times[sequence[i]-1,j-1] - position[i,j]
	return idle_time, position