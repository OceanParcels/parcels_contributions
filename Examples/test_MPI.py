try:
    from mpi4py import MPI
except:
    MPI = None

class ProcessorEntity():
    id = 0

if __name__ == '__main__':
    if MPI is None:
        exit()

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    id_list = []
    for i in range(0, mpi_size):
        id_list.append(i)

    pe = ProcessorEntity()
    partitions = mpi_comm.bcast(id_list, root=0)
    pe.id = partitions[mpi_rank]
    pe_accum = 0
    for i in range(0, mpi_size):
        pe_bcast = mpi_comm.bcast(pe.id, root=i)
        if i <= mpi_rank:
            pe_accum += pe_bcast

    mpi_comm.Barrier()
    print("PE {} - accum: {}".format(pe.id, pe_accum))