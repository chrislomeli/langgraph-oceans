1. The species/stock values won't match the join key. doc_chunks.species has to equal individuals.common_name, and those are Title-Case exact strings: Humpback Whale, Killer Whale, Blue Whale…
  So "humpback whale" (lowercase) and especially "orca" (line 53–54) will silently fail to join — the agent's species filter returns nothing. Fix the values to the exact common names ("orca" →
  "Killer Whale"). This is the #1 thing.

 **Chris** I see now what you were thinking.  I updated that based on what is in the database
  
  
2. The two killer-whale stocks are swapped/confused. killer-whale-enp-2024 is labeled "Eastern Pacific Southern Resident" (line 53) but the ENP file is the general stock — Southern Resident is
  the other file. Worth a glance.

**Chris** They look correct to me - based on the title of the document - not the file name
 
3. source lost its job. You set every doc's source to the same generic string (line 46). source was meant to be the citation handle — which SAR a chunk came from ("Humpback CA/OR/WA SAR").
  As-is, every chunk cites the same thing and per-doc provenance is gone. Either restore a per-doc source or derive a citation from species+stock+year.

**Chris** I did not get your intent here, but we already have the file name and we have the stock subject "Oregon California" - but we don't have anything to differentiate SARS from another kind of report.   
 I thought this was what you meant and I'm open, but I also think it makes more sense?


4. Keep/drop diverges from your own design. Your if subject: logic (line 178) keeps everything non-None, which keeps PBR and Net Productivity — but sar-collection-design.md says drop those
  (methodology; numbers live in the card). And KEEP_SECTIONS (line 76) is now dead code — nothing reads it. Decide consciously: either trust the design's keep-set, or update the design.

**Chris** I did not think we needed KEEP - we could use the None in KNOWN_HEADINGS instead.   
I had trouble understanding the shorthand title names in the design but i think i've modified it correctly now.


5. year is off by ~1 and inconsistent. humpback-caorwa-2021.pdf → year=2022 (line 51); several others are +1. Pick one definition (the filename/edition year matches the design and the
citation) and make them consistent.

**Chris** I updated the column name to `revised` .. this is the revised date on the document itself, not the file name.
