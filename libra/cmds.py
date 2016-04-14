from abc import ABCMeta
from abc import abstractmethod

from time import perf_counter

import logging

import numpy

import re

from .config import LibraConfig

from .params import ShParameter
from .params import LibraSimpleParameter
from .params import LibraOptionParameter


class SimpleOutputParser(metaclass=ABCMeta):

    """
    WRITEME
    """
    @abstractmethod
    def __call__(self, output_str):
        """
        Defines a function to parse and retrieve some (structured) output from a string
        """


class CopyOutputParser(SimpleOutputParser):

    def __call__(self, output_str):
        return str(output_str)


class ArrayOutputParser(SimpleOutputParser):

    def __call__(self, output_str):
        """
        Dirty parsing
        TODO: use a regex
        """
        #
        # split strings by newlines
        lines = output_str.split('\n')
        #
        # remove all the lines that are not numbers
        lls = []
        for ll in lines:
            try:
                lls.append(float(ll))
            except ValueError:
                pass
        #
        # convert to numpy array
        return numpy.array(lls)


class AverageStdParser(SimpleOutputParser):

    def __call__(self, output_str):
        """
        Eg.
            avg = -5.806689 +/- 0.021006
        """
        output_str = str(output_str)
        stats = re.findall(r"[-+]?\d*\.\d+|\d+", output_str)

        assert len(stats) == 2

        return [float(s) for s in stats]


class SimpleCommandWrapper(object):

    """
    A simple base wrapper around sh
    """

    def __init__(self,
                 cmd_name,
                 output_parser,
                 cmd_params=None,
                 print_output=False,
                 output_log=None):
        self.name = cmd_name
        self.out_parser = output_parser

        if cmd_params is None:
            cmd_params = []
        self.params = ShParameter.convert_params(cmd_params)

        self._print_output = print_output
        self._out_log = output_log
        self.output = None

        self.exec_time_ = None
        self.running_ = False

    def __call__(self):
        """
        Exectute the command
        """
        logging.debug('Executing "{} {} {}"'.format(LibraConfig.libra_cmd(),
                                                    self.name,
                                                    ' '.join(p for p in self.params)))
        self.running_ = True

        cmd_start_t = perf_counter()
        output = LibraConfig.libra_cmd()(self.name, *self.params)
        cmd_end_t = perf_counter()

        self.exec_time_ = cmd_end_t - cmd_start_t

        #
        # saving output
        if self._print_output:
            print(output)
        self.output = output.__repr__()
        if self._out_log:
            with open(self._out_log, 'w') as log:
                log.write(output)

        self.running_ = False
        #
        # parsing, then returning
        return self.out_parser(output)

    def __repr__(self):
        param_str = '\n'.join('\t{}'.format(v) for v in self.params)
        return 'libra {}\n{}'.format(self.name,
                                     param_str)

#
# LEARNING METHODS
#


class Cl(SimpleCommandWrapper):

    """
    Wrapper around the cl command (Chow-Liu Algorithm)
    """

    def __init__(self,
                 i,
                 o,
                 s=None,
                 prior=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('cl',
                         output_parser=CopyOutputParser(),
                         output_log=output_log,
                         print_output=print_output,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('s', s)])


class Bnlearn(SimpleCommandWrapper):

    """
    Wrapper around the bnlearn command (Bayesian Networks with tree CPTs)
    """

    def __init__(self,
                 i,
                 o,
                 s=None,
                 prior=None,
                 parents=None,
                 ps=None,
                 kappa=None,
                 psthresh=False,
                 maxs=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('bnlearn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('prior', prior),
                                     LibraSimpleParameter('parents', parents),
                                     LibraSimpleParameter('ps', ps),
                                     LibraSimpleParameter('kappa', kappa),
                                     LibraOptionParameter('psthresh', psthresh),
                                     LibraSimpleParameter('maxs', maxs), ])


class Acbn(SimpleCommandWrapper):

    """
    Wrapper around the acbn command (Bayesian Networks with Arithmetic Circuits)
    """

    def __init__(self,
                 i,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('acbn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('prior', prior),
                                     LibraSimpleParameter('parents', parents),
                                     LibraSimpleParameter('pe', pe),
                                     LibraOptionParameter('shrink', shrink),
                                     LibraSimpleParameter('ps', ps),
                                     LibraSimpleParameter('kappa', kappa),
                                     LibraOptionParameter('psthresh', psthresh),
                                     LibraSimpleParameter('maxe', maxs),
                                     LibraSimpleParameter('maxs', maxs),
                                     LibraOptionParameter('quick', quick),
                                     LibraOptionParameter('qgreedy', qgreedy),
                                     LibraSimpleParameter('freq', freq)])


class Acmn(SimpleCommandWrapper):

    """
    Wrapper around acmn command (Markov Networks with Arithmetic Circuits)
    """

    def __init__(self,
                 i,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('acmn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('sd', sd),
                                     LibraSimpleParameter('l1', l1),
                                     LibraSimpleParameter('pe', pe),
                                     LibraSimpleParameter('sloppy', sloppy),
                                     LibraOptionParameter('shrink', shrink),
                                     LibraSimpleParameter('ps', ps),
                                     LibraOptionParameter('psthresh', psthresh),
                                     LibraSimpleParameter('maxe', maxs),
                                     LibraSimpleParameter('maxs', maxs),
                                     LibraOptionParameter('quick', quick),
                                     LibraOptionParameter('halfsplit', halfsplit),
                                     LibraSimpleParameter('freq', freq)])


class Idspn(SimpleCommandWrapper):

    """
    Wrapper around idspn command (Sum-Product Networks with Arithmetic Circuits)
    """

    def __init__(self,
                 i,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('idspn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('l1', l1),
                                     LibraSimpleParameter('l', l),
                                     LibraSimpleParameter('ps', k),
                                     LibraSimpleParameter('sd', sd),
                                     LibraSimpleParameter('cp', cp),
                                     LibraSimpleParameter('vth', vth),
                                     LibraSimpleParameter('ext', ext),
                                     LibraSimpleParameter('ml1', minl1),
                                     LibraSimpleParameter('minedge', minedge),
                                     LibraSimpleParameter('minps', minps),
                                     LibraSimpleParameter('seed', seed),
                                     LibraOptionParameter('f', f)])


class Mtlearn(SimpleCommandWrapper):

    """
    Wrapper around the mtlearn command (Mixture of Chow-Liu Trees)
    """

    def __init__(self,
                 i,
                 o,
                 k,
                 s=None,
                 seed=None,
                 f=False,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('mtlearn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('k', k),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('seed', seed),
                                     LibraOptionParameter('f', f)])


class Dnlearn(SimpleCommandWrapper):

    """
    Wrapper around the dnlearn command (Dependency Networks)
    """

    def __init__(self,
                 i,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('dnlearn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('prior', prior),
                                     LibraSimpleParameter('ps', ps),
                                     LibraSimpleParameter('kappa', kappa),
                                     LibraOptionParameter('tree', tree),
                                     LibraOptionParameter('mincount', mincount),
                                     LibraOptionParameter('logistic', logistic),
                                     LibraSimpleParameter('l1', l1)])


class Dnboost(SimpleCommandWrapper):

    """
    Wrapper around the dnboost command (Dependency Networks with Logitboost)
    """

    def __init__(self,
                 i,
                 valid,
                 o,
                 numtrees,
                 numleaves,
                 s=None,
                 nu=None,
                 mincount=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('dnboost',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('valid', valid),
                                     LibraSimpleParameter('s', s),
                                     LibraSimpleParameter('numtrees', numtrees),
                                     LibraSimpleParameter('numleaves', numleaves),
                                     LibraSimpleParameter('nu', nu),
                                     LibraSimpleParameter('mincount', mincount)])


class Dn2mn(SimpleCommandWrapper):

    """
    Wrapper around dn2mn command (Markov Networks from Dependency Networks)
    """

    def __init__(self,
                 m,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('dn2mn',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraOptionParameter('base', base),
                                     LibraOptionParameter('bcounts', bcounts),
                                     LibraOptionParameter('marg', marg),
                                     LibraSimpleParameter('order', order),
                                     LibraOptionParameter('rev', rev),
                                     LibraOptionParameter('norev', norev),
                                     LibraOptionParameter('single', single),
                                     LibraOptionParameter('linear', linear),
                                     LibraOptionParameter('all', all),
                                     LibraSimpleParameter('maxlen', maxlen),
                                     LibraOptionParameter('uniq', uniq)])

#
# Weight learning
#


class Mnlearnw(SimpleCommandWrapper):

    """
    Wrapper around the mnlearnw command (Markow Network Weight learning)
    """

    def __init__(self,
                 m,
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
                 output_log=None):
        """
                 WRITEME
                 """
        super().__init__('mnlearnw',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('maxiter', maxiter),
                                     LibraSimpleParameter('sd', sd),
                                     LibraSimpleParameter('l1', l1),
                                     LibraOptionParameter('clib', clib),
                                     LibraOptionParameter('noclib', noclib),
                                     LibraOptionParameter('cache', cache),
                                     LibraOptionParameter('nocache', nocache)])


class Acopt(SimpleCommandWrapper):

    """
    Wrapper around the acopt command (Arithmetic Circuit Parameter learning)
    """

    def __init__(self,
                 m,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('acopt',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('ma', ma),
                                     LibraSimpleParameter('i', i),
                                     LibraSimpleParameter('o', o),
                                     LibraOptionParameter('gibbs', gibbs),
                                     LibraSimpleParameter('gc', gc),
                                     LibraSimpleParameter('gb', gb),
                                     LibraSimpleParameter('gs', gs),
                                     LibraOptionParameter('gdn', gdn),
                                     LibraOptionParameter('norb', norb),
                                     LibraSimpleParameter('seed', seed),
                                     LibraSimpleParameter('ev', ev),
                                     LibraSimpleParameter('init', init),
                                     LibraSimpleParameter('maxiter', maxiter)])


#
# Inference
#

class Acve(SimpleCommandWrapper):

    """
    Wrapper around the acve command
    (compiling a model into an Arithmetic Circuit via Variable Elimination)
    """

    def __init__(self,
                 m,
                 o,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('acve',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('o', o)])


class Acquery(SimpleCommandWrapper):

    """
    Wrapper around the acquery command (exact inference with Arithmetic Circuits)
    """

    def __init__(self,
                 m,
                 q,
                 ev=None,
                 sameev=False,
                 preprune=False,
                 marg=False,
                 mpe=False,
                 mo=False,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('acquery',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('sameev', sameev),
                                     LibraOptionParameter('preprune', preprune),
                                     LibraOptionParameter('marg', marg),
                                     LibraOptionParameter('mpe', mpe),
                                     LibraOptionParameter('mo', mo)])


class Spnquery(SimpleCommandWrapper):

    """
    Wrapper around the spnquery command (exact inference with Sum-Product Networks)
    """

    def __init__(self,
                 m,
                 q,
                 ev=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('spnquery',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev)])


class Mf(SimpleCommandWrapper):

    """
    Wrapper around the mf command (Mean Field approximate inference)
    """

    def __init__(self,
                 m,
                 q,
                 mo=None,
                 ev=None,
                 sameev=False,
                 maxiter=None,
                 thresh=None,
                 depnet=False,
                 roundrobin=False,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('mf',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('sameev', sameev),
                                     LibraSimpleParameter('maxiter', maxiter),
                                     LibraSimpleParameter('thresh', thresh),
                                     LibraOptionParameter('depnet', depnet),
                                     LibraOptionParameter('roundrobin', roundrobin)])


class Bp(SimpleCommandWrapper):

    """
    Wrapper around the bp command (Belief Propagation (approximate) inference)
    """

    def __init__(self,
                 m,
                 q,
                 mo=None,
                 ev=None,
                 sameev=False,
                 maxiter=None,
                 thresh=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('bp',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('sameev', sameev),
                                     LibraSimpleParameter('maxiter', maxiter),
                                     LibraSimpleParameter('thresh', thresh)])


class Maxprod(SimpleCommandWrapper):

    """
    Wrapper around the maxprod command (Max-Product (approximate) MPE inference)
    """

    def __init__(self,
                 m,
                 q,
                 mo=None,
                 ev=None,
                 sameev=False,
                 maxiter=None,
                 thresh=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('maxprod',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('sameev', sameev),
                                     LibraSimpleParameter('maxiter', maxiter),
                                     LibraSimpleParameter('thresh', thresh)])


class Gibbs(SimpleCommandWrapper):

    """
    Wrapper around the gibbs command (Gibbs sampling for approximate inference)
    """

    def __init__(self,
                 m,
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
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('gibbs',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('sameev', sameev),
                                     LibraOptionParameter('marg', marg),
                                     LibraSimpleParameter('speed', speed),
                                     LibraSimpleParameter('burnin', burnin),
                                     LibraSimpleParameter('sampling', sampling),
                                     LibraOptionParameter('norb', norb),
                                     LibraSimpleParameter('prior', prior),
                                     LibraSimpleParameter('seed', seed)])


class Icm(SimpleCommandWrapper):

    """
    Wrapper around the icm command (Iterated Conditional Modes approximate MPE inference)
    """

    def __init__(self,
                 m,
                 q,
                 mo=None,
                 ev=None,
                 sameev=False,
                 depnet=False,
                 restarts=None,
                 maxiter=None,
                 seed=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('icm',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('mo', mo),
                                     LibraSimpleParameter('q', q),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('sameev', sameev),
                                     LibraOptionParameter('depnet', depnet),
                                     LibraSimpleParameter('restarts', restarts),
                                     LibraSimpleParameter('maxiter', maxiter),
                                     LibraSimpleParameter('seed', seed)])


#
# Utilities
#


class Bnsample(SimpleCommandWrapper):

    """
    Wrapper around the bnsample command (sampling from a Bayesian Network)
    """

    def __init__(self,
                 m,
                 o,
                 n=None,
                 seed=None,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('bnsample',
                         output_parser=ArrayOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('n', n),
                                     LibraSimpleParameter('seed', seed)])


class Mscore(SimpleCommandWrapper):

    """
    Wrapper around the mscore command ((p)ll score for a model on some data)
    """

    def __init__(self,
                 m,
                 i,
                 depnet=False,
                 pll=False,
                 pervar=False,
                 clib=False,
                 noclib=False,
                 v=True,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        output_parser = None
        if v:
            output_parser = ArrayOutputParser
        else:
            output_parser = AverageStdParser

        super().__init__('mscore',
                         output_parser=output_parser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('i', i),
                                     LibraOptionParameter('depnet', depnet),
                                     LibraOptionParameter('pll', pll),
                                     LibraOptionParameter('pervar', pervar),
                                     LibraOptionParameter('clib', clib),
                                     LibraOptionParameter('noclib', noclib),
                                     LibraOptionParameter('v', v)])


class Mconvert(SimpleCommandWrapper):

    """
    Wrapper around the mconvert command (model conversion utility)
    """

    def __init__(self,
                 m,
                 o,
                 ev=None,
                 feat=False,
                 dn=False,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('mconvert',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('o', o),
                                     LibraSimpleParameter('ev', ev),
                                     LibraOptionParameter('feat', feat),
                                     LibraOptionParameter('dn', dn)])


class Spn2ac(SimpleCommandWrapper):

    """
    Wrapper around the spn2ac command (Sum-Product Network conversion to Arithmetic Circuits)
    """

    def __init__(self,
                 m,
                 o,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('spn2ac',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('m', m),
                                     LibraSimpleParameter('o', o)])


class Fstats(SimpleCommandWrapper):

    """
    Wrapper around the fstats command (file info)
    """

    def __init__(self,
                 i,
                 print_output=False,
                 output_log=None):
        """
        WRITEME
        """
        super().__init__('fstats',
                         output_parser=CopyOutputParser(),
                         print_output=print_output,
                         output_log=output_log,
                         cmd_params=[LibraSimpleParameter('i', i)])
