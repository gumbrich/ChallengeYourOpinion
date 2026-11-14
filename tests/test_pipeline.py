import csv
import tempfile
import unittest
from pathlib import Path
from scrape import append_headlines
from scripts.export_archive import read_source
from plot_frequencies import distr_nzz, distr_spiegel, distr_taz

class CorpusTests(unittest.TestCase):
    def test_deduplicate_normalize_preserve_unknown_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'news.csv'
            path.write_text('Teaser,Date\n" First  headline ",\n',encoding='utf-8')
            self.assertEqual(append_headlines(path,['First headline',' Second\nheadline ','Second headline',None], '2026-11-14'),1)
            with path.open(encoding='utf-8') as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['Date'],'')
            self.assertEqual(rows[1]['Date'],'2026-11-14')
            self.assertFalse(path.with_suffix('.csv.tmp').exists())
            exported=read_source(path,'example')
            self.assertEqual(exported[0]['headline'],'First headline')
            self.assertIsNone(exported[0]['capturedAt'])
            self.assertIsNone(exported[0]['articleUrl'])
            self.assertEqual(exported,read_source(path,'example'))
    def test_export_deduplicates_per_source_without_fabricating_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'focus.csv'
            path.write_text('Teaser\n" A headline "\nA headline\n\n',encoding='utf-8')
            rows=read_source(path,'focus')
            self.assertEqual(len(rows),1)
            self.assertIsNone(rows[0]['capturedAt'])
    def test_all_frequency_functions_return_local_counts(self):
        for function in (distr_nzz,distr_spiegel,distr_taz):
            self.assertGreater(sum(function().values()),0)

if __name__=='__main__': unittest.main()
