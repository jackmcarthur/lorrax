#!/usr/bin/env python3
"""Prepare the pinned SOC tutorial without modifying upstream inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ARCHIVE_SHA256 = 'fb84d8e26ba26c4e02772c1feb16fe6a2200087c6f3ef01198921c54c0e6c8af'
ARCHIVE_URL = ('https://drive.usercontent.google.com/download?'
               'id=15nj-0VcAPZr2TPSaGwgiYEOCgal5wkTb&export=download&confirm=t')
ORIGINAL_STAGES = ('2a-Wfn', '2b-Wfnq', '3a-Wfn_fi', '3b-Wfnq_fi', '4-Wfn_inteqp')
EXTRA_STAGES = ('2c-Wfn_full', '2d-Wfn_guarded')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize(archive, tutorial):
    """Extract only the checksum-pinned archive into a new working tree."""
    if tutorial.exists():
        raise FileExistsError(f'Refusing to replace an existing tutorial: {tutorial}')
    tutorial.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tutorial.parent) as temporary:
        temporary = Path(temporary)
        archive_origin = str(archive) if archive is not None else ARCHIVE_URL
        if archive is None:
            archive = temporary / '6a-SOC.tar.gz'
            with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output)
        if sha(archive) != ARCHIVE_SHA256:
            raise ValueError('Archive SHA256 differs from the pinned workshop download')
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                parts = Path(member.name).parts
                if (not parts or parts[0] != '6a-SOC' or '..' in parts
                        or Path(member.name).is_absolute()
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unexpected archive member: {member.name}')
            source.extractall(temporary)
        tree = temporary / '6a-SOC'
        # Explicit additive paths also work when this helper is invoked from an
        # already initialized tutorial. Never copy or replace upstream folders.
        overlay = Path(__file__).resolve().parent.parent
        additions = ('perlmutter', '2-gw/3-lorrax_fully_rel',
                     '3-bse/3-lorrax_fully_rel', '00-run-lorrax.sh', 'README-LORRAX.md',
                     'validate_tutorial.py')
        for name in additions:
            origin = overlay / name
            destination = tree / name
            if not origin.exists():
                raise FileNotFoundError(f'Incomplete LORRAX overlay: {origin}')
            if destination.exists():
                raise FileExistsError(f'Overlay would replace an upstream path: {name}')
            if origin.is_dir():
                shutil.copytree(origin, destination, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copy2(origin, destination)
        receipt = dict(archive=archive_origin, sha256=ARCHIVE_SHA256,
                       original_author='Bradford A. Barker, 2022', additions=list(additions),
                       note='Original README, LICENSE, physics inputs and pseudos preserved')
        (tree / 'perlmutter/ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        shutil.move(str(tree), tutorial)
    print(tutorial)


def prepare(tutorial, stage):
    """Generate extras, then share the original SCF density without replacing XML."""
    mf = tutorial / '1-mf/2-fully_rel'
    source = mf / '2a-Wfn'
    target = mf / stage
    scf = mf / '1-Density/MoS2.save'
    for required in ('data-file-schema.xml', 'charge-density.hdf5'):
        if not (scf / required).is_file():
            raise FileNotFoundError(f'Finish the original SCF first: {scf / required}')
    if stage in EXTRA_STAGES:
        original = (source / 'wfn.in').read_text()
        if original.count('nbnd = 200') != 1:
            raise ValueError('Unexpected original 2a-Wfn band input')
        if stage == '2c-Wfn_full':
            text = original.replace('nbnd = 200', 'nbnd = 40')
            text = text.split('K_POINTS crystal')[0] + 'K_POINTS crystal\n   9\n'
            text += ''.join(f'   {i/3:.15f} {j/3:.15f} 0.0 {1/9:.15f}\n'
                            for i in range(3) for j in range(3))
            converter = (source / 'wfn.pp.in').read_text().replace(
                'vxc_diag_nmax = 200', 'vxc_diag_nmax = 40')
            changes = ['nbnd 200 to40', 'Explicit full3x3 grid,9 equal-weight k-points']
            purpose = 'Full-coarse file-gauge interoperability for the native BSEMAT export'
        else:
            text = original.replace('nbnd = 200', 'nbnd = 204')
            converter = (source / 'wfn.pp.in').read_text()
            changes = ['nbnd 200 to204; no other NSCF or converter input changes']
            purpose = 'Four spare states for strict upper-edge checks of the 200-band fit'
        if target.exists():
            for filename, expected in (('wfn.in', text), ('wfn.pp.in', converter)):
                if (target / filename).read_text() != expected:
                    raise ValueError(f'Refusing to replace differing extra input: {target / filename}')
        else:
            target.mkdir()
            (target / 'wfn.in').write_text(text)
            (target / 'wfn.pp.in').write_text(converter)
            provenance = dict(purpose=purpose, source='2a-Wfn/wfn.in',
                              source_sha256=sha(source / 'wfn.in'), changes=changes,
                              fixed_potential='Original 1-Density/MoS2.save SCF charge',
                              unchanged='45 Ry, FR pseudos, cell, atomic positions, SOC Hamiltonian',
                              original_reference='2a-Wfn remains the untouched BGW reference',
                              usage='LORRAX sums and explicit zeta fit still stop at band200')
            (target / 'INPUT_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
            (target / 'README.extra.txt').write_text(
                purpose + '\n' + '\n'.join(changes) + '\n'
                'The original fixed SCF potential and basis are unchanged.\n'
                'Do not replace the original upstream 2a-Wfn reference.\n'
                'The extra204-band file supplies guard states; GW sums still use200 bands.\n')
    if not target.is_dir():
        raise FileNotFoundError(target)
    save = target / 'MoS2.save'
    save.mkdir(exist_ok=True)
    charge = save / 'charge-density.hdf5'
    if charge.is_symlink():
        if charge.resolve() != (scf / charge.name).resolve():
            raise ValueError(f'Unexpected charge-density link: {charge}')
    elif charge.exists():
        raise FileExistsError(f'Refusing to replace existing charge density: {charge}')
    else:
        charge.symlink_to('../../1-Density/MoS2.save/charge-density.hdf5')
    # A completed NSCF XML describes its own bands and k-points. Never overwrite it.
    schema = save / 'data-file-schema.xml'
    if not schema.exists():
        if (target / 'WFN').exists() or (target / 'wfn.out').exists():
            raise FileNotFoundError(f'Existing NSCF output has no schema; inspect manually: {target}')
        shutil.copy2(scf / schema.name, schema)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tutorial', type=Path,
                        default=Path(__file__).resolve().parent.parent)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Initialize a new tree from the checksum-pinned archive')
    init.add_argument('--archive', type=Path,
                      help='Use an existing pinned archive instead of downloading it')
    stage = commands.add_parser('stage', help='Prepare one NSCF stage after SCF; safe to repeat')
    stage.add_argument('name', choices=ORIGINAL_STAGES + EXTRA_STAGES)
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.archive.resolve() if args.archive else None, args.tutorial.resolve())
    else:
        prepare(args.tutorial.resolve(), args.name)


if __name__ == '__main__':
    main()
