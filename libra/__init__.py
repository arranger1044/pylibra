import os

import logging

import numpy

from .cmds import Cl
from .cmds import Bnlearn
from .cmds import Acbn
from .cmds import Acmn
from .cmds import Idspn
from .cmds import Mtlearn
from .cmds import Dnlearn
from .cmds import Dnboost
from .cmds import Dn2mn
from .cmds import Mnlearnw
from .cmds import Acopt
from .cmds import Acve
from .cmds import Mscore
from .cmds import Bnsample
from .cmds import Acquery
from .cmds import Spnquery
from .cmds import Mf
from .cmds import Bp
from .cmds import Maxprod
from .cmds import Gibbs
from .cmds import Icm
from .cmds import Mconvert
from .cmds import Spn2ac
from .cmds import Fstats

from .config import LibraConfig

from .utils import get_temp_file
from .utils import delete_temp_file

LIBRA_DONT_CARE_VAL = '*'

SCHEMA_EXT = '.schema'
DATA_EXT = '.data'
PARENTS_EXT = '.parents'
SAMPLE_EXT = '.samples'
EVIDENCE_EXT = '.ev'
QUERY_EXT = '.q'
AC_EXT = '.ac'
BN_EXT = '.bn'
MN_EXT = '.mn'

#
# input conversion routines
#


def check_schema(schema):
    """
    Check if schema is an already existing schema file
    or an iterable that needs to get serialized to a temporary file
    """
    if type(schema) == str and os.path.isfile(schema):
        return schema

    schema_array = numpy.array(schema, dtype='uint')
    schema_array = schema_array.reshape(1, schema_array.shape[0])

    schema_file = get_temp_file(SCHEMA_EXT)
    numpy.savetxt(schema_file, schema_array, delimiter=',', fmt='%u')

    return schema_file.name


def check_data(data):
    """
    Check if data is an already existing (training) data file
    or an iterable that needs to get serialized to a temporary file
    """
    if type(data) == str and os.path.isfile(data):
        return data

    data_array = numpy.array(data, dtype='uint')

    data_file = get_temp_file(DATA_EXT)
    numpy.savetxt(data_file, data_array, delimiter=',', fmt='%u')

    return data_file.name


def check_parents_file(parents):
    """
    Check if parents is an already existing file (specifying the parents constraints
    for a BN), or it is a dictionary in the form
        node_id : allowed_parents_ids
    """

    if type(parents) == str and os.path.isfile(parents):
        return parents

    parents_file = get_temp_file(PARENTS_EXT)
    for node_id, allowed_parents in parents.items():
        node_cons_str = None
        if allowed_parents:
            node_cons_str = '{}: none except {}\n'.format(node_id,
                                                          ' '.join(p for p in allowed_parents))
        else:
            node_cons_str = '{}: none{}\n'.format(node_id)
        parents_file.write(node_cons_str)

    parents_file.close()
    return parents_file.name


def check_evidence(evidence):
    """
    Check if evidence is an already existing file specifying evidences
    or it is an array-like containing evidences
    """

    if type(evidence) == str and os.path.isfile(evidence):
        return evidence

    evidence_file = get_temp_file(EVIDENCE_EXT)
    dont_care_val = LibraConfig.dont_care_val()
    logging.debug('Considering {} values as don\'t care'.format(dont_care_val))

    for ev in evidence:
        ev_str_values = []
        for val in ev:
            val_str = None
            if val == dont_care_val:
                val_str = LIBRA_DONT_CARE_VAL
            else:
                val_str = str(val)
            ev_str_values.append(val_str)
        ev_str = '{}\n'.format(','.join(ev_str_values))
        evidence_file.write(ev_str.encode('utf-8'))

    evidence_file.close()
    return evidence_file.name


def check_query(query):
    return check_evidence(query)


def cl(i,
       o,
       s=None,
       prior=None,
       print_output=False,
       output_log=None,
       delete_temp=False):
    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    #
    # create command wrapper
    cmd = Cl(i=i, o=o, s=s, prior=prior,
             print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)

    logging.info('cl finished ({} secs)'.format(cmd.exec_time_))
    return output


def bnlearn(i,
            o,
            s=None,
            prior=None,
            parents=None,
            ps=None,
            kappa=None,
            psthresh=False,
            maxs=None,
            print_output=False,
            output_log=None,
            delete_temp=False):
    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    if parents is not None:
        parents = check_parents_file(parents)

    #
    # create command wrapper
    cmd = Bnlearn(i=i, o=o, s=s, prior=prior,
                  parents=parents, ps=ps, kappa=kappa,
                  psthresh=psthresh, maxs=maxs,
                  print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)
        delete_temp_file(parents)

    logging.info('bnlearn finished ({} secs)'.format(cmd.exec_time_))
    return output


def acbn(i,
         o,
         mo,
         s=None,
         prior=None,
         parents=None,
         pe=None,
         shrink=False,
         ps=None,
         kappa=None,
         psthresh=False,
         maxe=None,
         maxs=None,
         quick=False,
         qgreedy=False,
         freq=None,
         print_output=False,
         output_log=None,
         delete_temp=False):
    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    if parents is not None:
        parents = check_parents_file(parents)

    if mo is None:
        pre, ext = os.path.splitext(o)
        mo = pre + BN_EXT

    #
    # create command wrapper
    cmd = Acbn(i=i, o=o, mo=mo, s=s,
               prior=prior, parents=parents, pe=pe,
               shrink=shrink, ps=ps, kappa=kappa,
               psthresh=psthresh, maxe=maxe, maxs=maxs,
               quick=quick, qgreedy=qgreedy, freq=freq,
               print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)
        delete_temp_file(parents)

    logging.info('acbn finished ({} secs)'.format(cmd.exec_time_))
    return output


def acmn(i,
         o,
         mo,
         s=None,
         sd=None,
         l1=None,
         pe=None,
         sloppy=None,
         shrink=False,
         ps=None,
         psthresh=False,
         maxe=None,
         maxs=None,
         quick=False,
         halfsplit=False,
         freq=None,
         print_output=False,
         output_log=None,
         delete_temp=False):
    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    if mo is None:
        pre, ext = os.path.splitext(o)
        mo = pre + MN_EXT

    #
    # create command wrapper
    cmd = Acmn(i=i, o=o, mo=mo, s=s,
               sd=sd, l1=l1, pe=pe,
               sloppy=sloppy, shrink=shrink, ps=ps,
               psthresh=psthresh, maxe=maxe, maxs=maxs,
               quick=quick, halfsplit=halfsplit, freq=freq,
               print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)

    logging.info('acmn finished ({} secs)'.format(cmd.exec_time_))
    return output


def idspn(i,
          o,
          s=None,
          l1=None,
          l=None,
          ps=None,
          k=None,
          sd=None,
          cp=None,
          vth=None,
          ext=None,
          minl1=None,
          minedge=None,
          minps=None,
          seed=None,
          f=False,
          print_output=False,
          output_log=None,
          delete_temp=False):
    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    #
    # create command wrapper
    cmd = Idspn(i=i, o=o,  s=s,
                l1=l1, l=l, ps=ps,
                k=k, sd=sd, cp=cp,
                vth=vth, ext=ext, minl1=minl1,
                minedge=minedge, minps=minps,
                seed=seed, f=f,
                print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)

    logging.info('idspn finished ({} secs)'.format(cmd.exec_time_))
    return output


def mtlearn(i,
            o,
            k,
            s=None,
            seed=None,
            f=False,
            print_output=False,
            output_log=None,
            delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    #
    # create command wrapper
    cmd = Mtlearn(i=i, o=o, k=k,  s=s,
                  seed=seed, f=f,
                  print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)

    logging.info('mtlearn finished ({} secs)'.format(cmd.exec_time_))
    return output


def dnlearn(i,
            o,
            s=None,
            prior=None,
            ps=None,
            kappa=None,
            tree=False,
            mincount=None,
            logistic=False,
            l1=None,
            print_output=False,
            output_log=None,
            delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    if s is not None:
        s = check_schema(s)

    #
    # create command wrapper
    cmd = Dnlearn(i=i, o=o,  s=s,
                  prior=prior, ps=ps, kappa=kappa,
                  tree=tree, mincount=mincount, logistic=logistic,
                  l1=l1,
                  print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(s)

    logging.info('dnlearn finished ({} secs)'.format(cmd.exec_time_))
    return output


def dnboost(i,
            valid,
            o,
            numtrees,
            numleaves,
            s=None,
            nu=None,
            mincount=None,
            print_output=False,
            output_log=None,
            delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    if valid is not None:
        valid = check_data(valid)

    if s is not None:
        s = check_schema(s)

    #
    # create command wrapper
    cmd = Dnboost(i=i, o=o,  valid=valid, s=s,
                  numtrees=numtrees, numleaves=numleaves, nu=nu,
                  mincount=mincount,
                  print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)
        delete_temp_file(valid)
        delete_temp_file(s)

    logging.info('dnlearn finished ({} secs)'.format(cmd.exec_time_))
    return output


def dn2mn(m,
          i,
          o,
          base=False,
          bcounts=False,
          marg=False,
          order=None,
          rev=False,
          norev=False,
          single=False,
          linear=False,
          all=False,
          maxlen=None,
          uniq=False,
          print_output=False,
          output_log=None,
          delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    #
    # create command wrapper
    cmd = Dn2mn(m=m, i=i, o=o,
                base=base, bcounts=bcounts, marg=marg,
                order=order, rev=rev, norev=norev,
                single=single, linear=linear, all=all,
                maxlen=maxlen, uniq=uniq,
                print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)

    logging.info('dn2mn finished ({} secs)'.format(cmd.exec_time_))
    return output


def mnlearnw(m,
             i,
             o,
             maxiter=None,
             sd=None,
             l1=None,
             clib=True,
             noclib=False,
             cache=True,
             nocache=False,
             print_output=False,
             output_log=None,
             delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    #
    # create command wrapper
    cmd = Mnlearnw(m=m, i=i, o=o,
                   maxiter=maxiter, sd=sd, l1=l1,
                   clib=clib, noclib=noclib, cache=cache,
                   nocache=nocache,
                   print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)

    logging.info('mnlearnw finished ({} secs)'.format(cmd.exec_time_))
    return output


def acopt(m,
          ma,
          i,
          o,
          gibbs=False,
          gspeed=None,
          gc=None,
          gb=None,
          gs=None,
          gdn=False,
          norb=False,
          seed=None,
          ev=None,
          init=None,
          maxiter=None,
          print_output=False,
          output_log=None,
          delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    #
    # TODO: check init and ev

    #
    # TODO: check gspeed

    #
    # create command wrapper
    cmd = Acopt(m=m, ma=ma, i=i, o=o,
                gibbs=gibbs, gspeed=gspeed,
                gc=gc, gb=gb, gs=gs,
                gdn=gdn, norb=norb, seed=seed,
                ev=ev, init=init, maxiter=maxiter,
                print_output=print_output, output_log=output_log)
    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)

    logging.info('acopt finished ({} secs)'.format(cmd.exec_time_))
    return output


def bnsample(m,
             o=None,
             n=None,
             seed=None,
             print_output=False,
             output_log=None,
             delete_temp=False):

    #
    # the output o is optional
    if o is None:
        temp_sample_file = get_temp_file(SAMPLE_EXT)
        o = temp_sample_file.name
        temp_sample_file.close()

    #
    # create command wrapper
    cmd = Bnsample(m=m, o=o,
                   n=n, seed=seed,
                   print_output=print_output, output_log=output_log)

    #
    # executing
    cmd()

    #
    # retrieving output from o
    output = numpy.loadtxt(o, dtype='int', delimiter=',')

    n_samples = n if n is not None else 1000
    assert output.shape[0] == n_samples

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(o)

    logging.info('bnsample finished ({} secs)'.format(cmd.exec_time_))
    return output


def mscore(m,
           i,
           depnet=False,
           pll=False,
           pervar=False,
           clib=False,
           noclib=False,
           v=True,
           print_output=False,
           output_log=None,
           delete_temp=False):

    #
    # check for inputs
    i = check_data(i)

    #
    # create command wrapper
    cmd = Mscore(m=m, i=i,
                 depnet=depnet, pll=pll, pervar=pervar,
                 clib=clib, noclib=noclib, v=v,
                 print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(i)

    logging.info('mscore finished ({} secs)'.format(cmd.exec_time_))
    return output


def acve(m,
         o,
         print_output=False,
         output_log=None):

    #
    # create command wrapper
    cmd = Acve(m=m, o=o,
               print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    logging.info('acve finished ({} secs)'.format(cmd.exec_time_))
    return output


def acquery(m,
            q,
            ev=None,
            sameev=False,
            preprune=False,
            marg=False,
            mpe=False,
            mo=False,
            print_output=False,
            output_log=None,
            delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Acquery(m=m, q=q, ev=ev,
                  sameev=sameev, preprune=preprune, marg=marg,
                  mpe=mpe, mo=mo,
                  print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('acquery finished ({} secs)'.format(cmd.exec_time_))
    return output


def spnquery(m,
             q,
             ev=None,
             print_output=False,
             output_log=None,
             delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Spnquery(m=m, q=q, ev=ev,
                   print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('spnquery finished ({} secs)'.format(cmd.exec_time_))
    return output


def mf(m,
       q,
       mo=None,
       ev=None,
       sameev=False,
       maxiter=None,
       thresh=None,
       depnet=False,
       roundrobin=False,
       print_output=False,
       output_log=None,
       delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Mf(m=m, q=q, ev=ev, mo=mo,
             sameev=sameev, maxiter=maxiter, thresh=thresh,
             depnet=depnet, roundrobin=roundrobin,
             print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('mf finished ({} secs)'.format(cmd.exec_time_))
    return output


def bp(m,
       q,
       mo=None,
       ev=None,
       sameev=False,
       maxiter=None,
       thresh=None,
       print_output=False,
       output_log=None,
       delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Bp(m=m, q=q, ev=ev, mo=mo,
             sameev=sameev, maxiter=maxiter, thresh=thresh,
             print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('bp finished ({} secs)'.format(cmd.exec_time_))
    return output


def maxprod(m,
            q,
            mo=None,
            ev=None,
            sameev=False,
            maxiter=None,
            thresh=None,
            print_output=False,
            output_log=None,
            delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Maxprod(m=m, q=q, ev=ev, mo=mo,
                  sameev=sameev, maxiter=maxiter, thresh=thresh,
                  print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('maxprod finished ({} secs)'.format(cmd.exec_time_))
    return output


def gibbs(m,
          q,
          mo=None,
          so=None,
          ev=None,
          sameev=False,
          marg=False,
          speed=None,
          chains=None,
          burnin=None,
          sampling=None,
          norb=False,
          prior=None,
          seed=None,
          print_output=False,
          output_log=None,
          delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Gibbs(m=m, q=q, ev=ev,
                mo=mo, so=so,
                sameev=sameev, marg=marg,
                speed=speed, chains=chains, burnin=burnin,
                sampling=sampling, norb=norb, prior=prior,
                seed=seed,
                print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('gibbs finished ({} secs)'.format(cmd.exec_time_))
    return output


def icm(m,
        q,
        mo=None,
        ev=None,
        sameev=False,
        depnet=False,
        restarts=None,
        maxiter=None,
        seed=None,
        print_output=False,
        output_log=None,
        delete_temp=False):

    #
    # check for inputs
    if q is not None:
        q = check_query(q)

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Icm(m=m, q=q, ev=ev,
              mo=mo,
              sameev=sameev, depnet=depnet,
              restarts=restarts, maxiter=maxiter, seed=seed,
              print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(q)
        delete_temp_file(ev)

    logging.info('icm finished ({} secs)'.format(cmd.exec_time_))
    return output


def mconvert(m,
             o,
             ev=None,
             feat=False,
             dn=False,
             print_output=False,
             output_log=None,
             delete_temp=False):

    if ev is not None:
        ev = check_evidence(ev)

    #
    # create command wrapper
    cmd = Mconvert(m=m, ev=ev, o=o,
                   feat=feat, dn=dn,
                   print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    #
    # cleaning, if necessary
    if delete_temp:
        delete_temp_file(ev)

    logging.info('mconvert finished ({} secs)'.format(cmd.exec_time_))
    return output


def spn2ac(m,
           o,
           print_output=False,
           output_log=None):

    #
    # create command wrapper
    cmd = Mconvert(m=m, o=o,
                   print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    logging.info('spn2ac finished ({} secs)'.format(cmd.exec_time_))
    return output


def fstats(i,
           print_output=False,
           output_log=None):
    #
    # create command wrapper
    cmd = Fstats(i=i,
                 print_output=print_output, output_log=output_log)

    #
    # executing
    output = cmd()

    logging.info('fstats finished ({} secs)'.format(cmd.exec_time_))
    return output
