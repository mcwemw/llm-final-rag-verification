# eval_40pos_v1.1 — hallucinated claims for review (53)
Each row: claim (from the response) and the RAGTruth human span(s) it overlaps. `overlap` = fraction of the claim's characters covered by a span. Rows with overlap < 0.3 are marked ⚠ (claim only touches the span edge).

## 1. `ragtruth-11908-c2` — UNSOURCED, overlap 0.3676
- model: llama-2-13b-chat
- claim: The specific amount of pay varies by location, with the highest average pay in Alaska ($23.70 per hour or $49,400 per year) and the lowest average pay in Mississippi ($18.60 per hour or $38,900 per year).
- span (Evident Baseless Info): the lowest average pay in Mississippi ($18.60 per hour or $38,900 per year)
  - annotator note: HIGH INTRO OF NEW INFO / It includes information about the lowest average pay of automotive technicians in Mississippi ($18.60 per hour or $38,900 per year), which is not mentioned in the original source.

## 2. `ragtruth-12460-c3` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: Indianapolis receives an average of 17 inches of snow per year, with the most snow falling in January (7.3 inches).
- span (Evident Baseless Info): Indianapolis receives an average of 17 inches of snow per year, with the most snow falling in January (7.3 inches). Fort Wayne receives an average of 35 inches of snow per year, with the most snow falling in February (12.3 inches). Gary receives an average of 24 inches of snow per year, with the most snow falling in January (9.5 inches). South Bend receives an average of 31 inches of snow per year, with the most snow falling in February (10.5 inches).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / This was not mentioned in the passages and original answer

## 3. `ragtruth-12460-c4` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: Fort Wayne receives an average of 35 inches of snow per year, with the most snow falling in February (12.3 inches).
- span (Evident Baseless Info): Indianapolis receives an average of 17 inches of snow per year, with the most snow falling in January (7.3 inches). Fort Wayne receives an average of 35 inches of snow per year, with the most snow falling in February (12.3 inches). Gary receives an average of 24 inches of snow per year, with the most snow falling in January (9.5 inches). South Bend receives an average of 31 inches of snow per year, with the most snow falling in February (10.5 inches).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / This was not mentioned in the passages and original answer

## 4. `ragtruth-12460-c5` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: Gary receives an average of 24 inches of snow per year, with the most snow falling in January (9.5 inches).
- span (Evident Baseless Info): Indianapolis receives an average of 17 inches of snow per year, with the most snow falling in January (7.3 inches). Fort Wayne receives an average of 35 inches of snow per year, with the most snow falling in February (12.3 inches). Gary receives an average of 24 inches of snow per year, with the most snow falling in January (9.5 inches). South Bend receives an average of 31 inches of snow per year, with the most snow falling in February (10.5 inches).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / This was not mentioned in the passages and original answer

## 5. `ragtruth-12460-c6` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: South Bend receives an average of 31 inches of snow per year, with the most snow falling in February (10.5 inches).
- span (Evident Baseless Info): Indianapolis receives an average of 17 inches of snow per year, with the most snow falling in January (7.3 inches). Fort Wayne receives an average of 35 inches of snow per year, with the most snow falling in February (12.3 inches). Gary receives an average of 24 inches of snow per year, with the most snow falling in January (9.5 inches). South Bend receives an average of 31 inches of snow per year, with the most snow falling in February (10.5 inches).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / This was not mentioned in the passages and original answer

## 6. `ragtruth-12460-c8` — UNSOURCED, overlap 0.2917 ⚠
- model: llama-2-13b-chat
- claim: The summer months are hot and humid, with temperatures ranging from 70°F to 90°F (21°C to 32°C).
- span (Evident Baseless Info): 70°F to 90°F (21°C to 32°C).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / "70°F to 90°F (21°C to 32°C)." was not mentioned in the passages and original answer

## 7. `ragtruth-12460-c9` — UNSOURCED, overlap 0.2755 ⚠
- model: llama-2-13b-chat
- claim: The winter months are chilly and windy, with temperatures ranging from 15°F to 35°F (-9°C to 1°C).
- span (Evident Baseless Info): 15°F to 35°F (-9°C to 1°C). The year-round average temperature is around 50°F (10°C).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / "15°F to 35°F (-9°C to 1°C). The year-round average temperature is around 50°F (10°C)." was not mentioned in the passages and original answer

## 8. `ragtruth-12460-c10` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: The year-round average temperature is around 50°F (10°C).
- span (Evident Baseless Info): 15°F to 35°F (-9°C to 1°C). The year-round average temperature is around 50°F (10°C).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / "15°F to 35°F (-9°C to 1°C). The year-round average temperature is around 50°F (10°C)." was not mentioned in the passages and original answer

## 9. `ragtruth-12518-c2` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: For example, the digestive system breaks down food into smaller molecules that can be absorbed by the body
- span (Subtle Baseless Info): For example, the digestive system breaks down food into smaller molecules that can be absorbed by the body, while the circulatory system transports nutrients and oxygen throughout the body. Similarly, the nervous system controls and coordinates the activities of all other systems in the body. Overall, these systems work together to maintain homeostasis and ensure the proper functioning of the body as a whole.
  - annotator note: LOW INTRO OF NEW INFO / This may be true but the function of digestive system and nervous system are not mentioned in the original source.

## 10. `ragtruth-12518-c3` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: the circulatory system transports nutrients and oxygen throughout the body.
- span (Subtle Baseless Info): For example, the digestive system breaks down food into smaller molecules that can be absorbed by the body, while the circulatory system transports nutrients and oxygen throughout the body. Similarly, the nervous system controls and coordinates the activities of all other systems in the body. Overall, these systems work together to maintain homeostasis and ensure the proper functioning of the body as a whole.
  - annotator note: LOW INTRO OF NEW INFO / This may be true but the function of digestive system and nervous system are not mentioned in the original source.

## 11. `ragtruth-12518-c4` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: Similarly, the nervous system controls and coordinates the activities of all other systems in the body.
- span (Subtle Baseless Info): For example, the digestive system breaks down food into smaller molecules that can be absorbed by the body, while the circulatory system transports nutrients and oxygen throughout the body. Similarly, the nervous system controls and coordinates the activities of all other systems in the body. Overall, these systems work together to maintain homeostasis and ensure the proper functioning of the body as a whole.
  - annotator note: LOW INTRO OF NEW INFO / This may be true but the function of digestive system and nervous system are not mentioned in the original source.

## 12. `ragtruth-12518-c5` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: Overall, these systems work together to maintain homeostasis and ensure the proper functioning of the body as a whole.
- span (Subtle Baseless Info): For example, the digestive system breaks down food into smaller molecules that can be absorbed by the body, while the circulatory system transports nutrients and oxygen throughout the body. Similarly, the nervous system controls and coordinates the activities of all other systems in the body. Overall, these systems work together to maintain homeostasis and ensure the proper functioning of the body as a whole.
  - annotator note: LOW INTRO OF NEW INFO / This may be true but the function of digestive system and nervous system are not mentioned in the original source.

## 13. `ragtruth-12602-c3` — FAIL, overlap 1.0
- model: mistral-7B-instruct
- claim: However, the Italian meaning of Teresa is not explicitly stated in the passages.
- span (Evident Conflict): However, the Italian meaning of Teresa is not explicitly stated in the passages.
  - annotator note: EVIDENT CONFLICT / Original: In Italian the meaning of the name Teresa is: Harvester / Generative: "...the Italian meaning of Teresa is not explicitly stated in the passages."

## 14. `ragtruth-12765-c4` — FAIL, overlap 0.1341 ⚠
- model: llama-2-7b-chat
- claim: If the assessee is entitled to claim depreciation on the residual value of a discarded asset, they can do so even if the asset is not used for business or professional purposes in the current year or subsequent years, as per Section 50(1) of the Income Tax Act.
- span (Evident Conflict): Section 50(1) of the Income Tax Act
  - annotator note: EVIDENT CONFLICT / Original: section 43(6) / Generated: Section 50(1) of the Income Tax Act

## 15. `ragtruth-13105-c5` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Draw a horizontal line for the x-axis and a vertical line for the y-axis.
- span (Evident Baseless Info): 4. Draw a horizontal line for the x-axis and a vertical line for the y-axis.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *no mention* / Generative: 4. Draw a horizontal line for the x-axis and a vertical line for the y-axis.

## 16. `ragtruth-13105-c6` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Label the x-axis with the categories or groups you are comparing.
- span (Evident Baseless Info): 5. Label the x-axis with the categories or groups you are comparing.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *no mention* / Generative: 5. Label the x-axis with the categories or groups you are comparing.

## 17. `ragtruth-13105-c7` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Label the y-axis with the appropriate scale that represents the values or measurements.
- span (Evident Baseless Info): 6. Label the y-axis with the appropriate scale that represents the values or measurements.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *no mention* / Generative: 6. Label the y-axis with the appropriate scale that represents the values or measurements.

## 18. `ragtruth-13105-c9` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Use different colors or patterns to distinguish between different categories or groups.
- span (Evident Baseless Info): 8. Use different colors or patterns to distinguish between different categories or groups.
9. Add a legend if necessary to explain the colors or patterns used.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *no mention of use of pattern* / Generative: 8. Use different colors or patterns to distinguish between different categories or groups. / 9. Add a legend if necessary to explain the colors or patterns used.

## 19. `ragtruth-13105-c10` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Add a legend if necessary to explain the colors or patterns used.
- span (Evident Baseless Info): 8. Use different colors or patterns to distinguish between different categories or groups.
9. Add a legend if necessary to explain the colors or patterns used.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *no mention of use of pattern* / Generative: 8. Use different colors or patterns to distinguish between different categories or groups. / 9. Add a legend if necessary to explain the colors or patterns used.

## 20. `ragtruth-13136-c6` — UNSOURCED, overlap 0.2595 ⚠
- model: mistral-7B-instruct
- claim: Here, you can set the sleep time by selecting the appropriate option from the drop-down menu or entering a custom value in seconds.
- span (Evident Baseless Info): entering a custom value in seconds
  - annotator note: HIGH INTRO OF NEW INFO / Entering a custom value in seconds is not mentioned in the original sources.

## 21. `ragtruth-13136-c7` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: After making your changes, click "Apply" and then "OK" to save them.
- span (Evident Baseless Info): 6. After making your changes, click "Apply" and then "OK" to save them.
  - annotator note: HIGH INTRO OF NEW INFO / It includes additional steps such as clicking 'Apply' and then 'OK' to save the changes, which are not mentioned in the original sources.

## 22. `ragtruth-13155-c2` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Improves mood and reduces stress: According to passage 1, hugging has a positive impact on the body's physiology, resulting in feelings of relaxation and reduced stress levels.
- span (Subtle Baseless Info): 1. Improves mood and reduces stress: According to passage 1, hugging has a positive impact on the body's physiology, resulting in feelings of relaxation and reduced stress levels.
  - annotator note: LOW INTRODUCTION OF NEW INFORMATION / Original: *not specifically stated* / Generative: 1. Improves mood and reduces stress: According to passage 1, hugging has a positive impact on the body's physiology, resulting in feelings of relaxation and reduced stress levels.

## 23. `ragtruth-13155-c5` — UNSOURCED, overlap 0.1739 ⚠
- model: llama-2-7b-chat
- claim: Reduces anxiety and worry: Study published in Psychological Science (referenced in passage 3) shows that hugging and touch can significantly reduce anxiety and worries about mortality.
- span (Evident Baseless Info): can significantly reduce anxiety
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *no mention of use of anxiety* / Generative: can significantly reduce anxiety

## 24. `ragtruth-13155-c6` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Increases feelings of social connection: Hugging can create a sense of belonging and social connection, which is essential for mental well-being (Passage 1).
- span (Evident Baseless Info): 4. Increases feelings of social connection: Hugging can create a sense of belonging and social connection, which is essential for mental well-being (Passage 1).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned in passage 1* / Generative: 4. Increases feelings of social connection: Hugging can create a sense of belonging and social connection, which is essential for mental well-being (Passage 1).

## 25. `ragtruth-13155-c7` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Supports emotional healing: The physical contact involved in hugging can help release oxytocin, a hormone that promotes emotional healing and bonding (Passage 1).
- span (Evident Baseless Info): 5. Supports emotional healing: The physical contact involved in hugging can help release oxytocin, a hormone that promotes emotional healing and bonding (Passage 1).
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned in passage 1* / Generative: 5. Supports emotional healing: The physical contact involved in hugging can help release oxytocin, a hormone that promotes emotional healing and bonding (Passage 1).

## 26. `ragtruth-13363-c11` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Make sure to dry them completely to avoid any stains or spots.
- span (Subtle Baseless Info): 10. Make sure to dry them completely to avoid any stains or spots.
  - annotator note: LOW INTRODUCTION OF NEW INFORMATION / Original: *not mentioned* / Generative: 10. Make sure to dry them completely to avoid any stains or spots

## 27. `ragtruth-14007-c6` — UNSOURCED, overlap 0.1017 ⚠
- model: llama-2-7b-chat
- claim: It is introduced by a relative pronoun (who, whose, him, her, it, that) or a subordinate conjunction (when and where).
- span (Evident Baseless Info): him, her, it
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Passages did not provide pronouns such as him, her, and it, as relative pronouns.

## 28. `ragtruth-14131-c1` — UNSOURCED, overlap 0.2348 ⚠
- model: gpt-3.5-turbo-0613
- claim: Examples of potential energy include a car parked on a hill, a ball at the top of a ramp, and a satellite in orbit.
- span (Evident Baseless Info): a ball at the top of a ramp
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *this is not mentioned* / Generative: a ball at the top of a ramp

## 29. `ragtruth-14293-c3` — UNSOURCED, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: A higher carbon footprint contributes to climate change and global warming.
- span (Subtle Baseless Info): . A higher carbon footprint contributes to climate change and global warming.
  - annotator note: LOW INTRO OF NEW INFO / Passages have no mention of this but may stand to be accurate and fits within the context of the question and passages.

## 30. `ragtruth-14323-c4` — FAIL, overlap 1.0
- model: gpt-3.5-turbo-0613
- claim: Overall, patterned rollers have been found to stimulate the skin and provide a healthy glow, as mentioned in passage 2.
- span (Evident Conflict): Overall, patterned rollers have been found to stimulate the skin and provide a healthy glow, as mentioned in passage 2.
  - annotator note: EVIDENT CONFLICT / Original: "I have been using the Dermaroller... It seems like it is stimulating my skin..." / Generative: "Overall, patterned rollers have been found to stimulate the skin and provide a healthy glow, as mentioned in passage 2."

## 31. `ragtruth-14806-c4` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: "Today's temperature in Bucharest is 22°C (72°F), humidity is 82%, wind speed is 6 mph (9 km/h)."
- span (Evident Baseless Info): "Today's temperature in Bucharest is 22°C (72°F), humidity is 82%, wind speed is 6 mph (9 km/h)."
  - annotator note: HIGH INTRO OF NEW INFO / Original: No mention of the current temperature in Bucharest / Generative: "Today's temperature in Bucharest is 22°C (72°F), humidity is 82%, wind speed is 6 mph (9 km/h)."

## 32. `ragtruth-14806-c5` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: Therefore, the current temperature in Bucharest is 22°C (72°F).
- span (Evident Baseless Info): Therefore, the current temperature in Bucharest is 22°C (72°F).
  - annotator note: HIGH INTRO OF NEW INFO / Original: No mention of the current temperature in Bucharest / Generative: "Therefore, the current temperature in Bucharest is 22°C (72°F)."

## 33. `ragtruth-14846-c10` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: Note: Depending on the recipe, you may want to adjust the cooking time or ingredient proportions accordingly.
- span (Evident Baseless Info): Note: Depending on the recipe, you may want to adjust the cooking time or ingredient proportions accordingly. Additionally, you can use any type of bread or mix in other ingredients like cheese or herbs for extra flavor.
  - annotator note: HIGH INTRO OF NEW INFO / No mention of this additional detail.

## 34. `ragtruth-14846-c11` — UNSOURCED, overlap 1.0
- model: mistral-7B-instruct
- claim: Additionally, you can use any type of bread or mix in other ingredients like cheese or herbs for extra flavor.
- span (Evident Baseless Info): Note: Depending on the recipe, you may want to adjust the cooking time or ingredient proportions accordingly. Additionally, you can use any type of bread or mix in other ingredients like cheese or herbs for extra flavor.
  - annotator note: HIGH INTRO OF NEW INFO / No mention of this additional detail.

## 35. `ragtruth-15010-c2` — FAIL, overlap 0.9932
- model: llama-2-13b-chat
- claim: Monozygotic (identical) twins occur when the split takes place after the ninth day after fertilization, specifically within 8 days of fertilization.
- span (Evident Conflict): Monozygotic (identical) twins occur when the split takes place after the ninth day after fertilization
  - annotator note: EVIDENT CONFLICT: / Original: "Monoamniotic" twins occur when the split takes place after the ninth day after fertilization. / AIGC: "Monozygotic (identical)" twins occur when the split takes place after the ninth day after fertilization
- span (Evident Conflict): Monozygotic (identical) twins occur when the split takes place after the ninth day after fertilization, specifically within 8 days of fertilization
  - annotator note: EVIDENT CONFLICT: / Original: "Monoamniotic" twins occur when the split takes place after the ninth day after fertilization and within 8 days of fertilization, it is actually Conjoined twins that are formed. / AIGC: Monozygotic (identical) twins occur when the split takes place after the ninth day after fertilization, specifically within 8 days of fertilization

## 36. `ragtruth-15677-c4` — UNSOURCED, overlap 0.8814
- model: llama-2-70b-chat
- claim: Poor hygiene: Not properly cleaning the skin can lead to an accumulation of oil and dirt, making the skin appear oily.
- span (Subtle Baseless Info): Not properly cleaning the skin can lead to an accumulation of oil and dirt, making the skin appear oily.
  - annotator note: LOW INTRODUCTION OF NEW INFORMATION / Passages did not mention not properly cleaning the skin as example of poor hygiene.

## 37. `ragtruth-15677-c5` — UNSOURCED, overlap 0.8346
- model: llama-2-70b-chat
- claim: Topical applications: Using products that are too harsh or contain heavy oils can irritate the skin and cause it to produce more oil.
- span (Subtle Baseless Info): Using products that are too harsh or contain heavy oils can irritate the skin and cause it to produce more oil.
  - annotator note: LOW INTRODUCTION OF NEW INFORMATION / Passages did not mention the use of too harsh products irritating the skin and causing to produce more oils.

## 38. `ragtruth-15688-c3` — UNSOURCED, overlap 0.302
- model: llama-2-13b-chat
- claim: High humidity: The climate in the tropical rainforest is very humid due to the high levels of rainfall, with an average humidity level of 70% to 90%.
- span (Evident Baseless Info): with an average humidity level of 70% to 90%.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned* / Generative: "with an average humidity level of 70% to 90%."

## 39. `ragtruth-15735-c5` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Average temperature in Tallahassee, Florida: 79 °F (26 °C) - 56 °F (13 °C) = 23 °F (12 °C)
- span (Subtle Baseless Info): Average temperature in Tallahassee, Florida: 79 °F (26 °C) - 56 °F (13 °C) = 23 °F (12 °C)
  - annotator note: LOW INTRO OF NEW INFO / - Passage 1 mentioned 79 degrees F (ave. high temp) and 56 degrees F(ave. low temp). However, AI added information specifically "23 degrees F "( difference of average high temperature and low temperature.)

## 40. `ragtruth-15755-c5` — UNSOURCED, overlap 1.0
- model: llama-2-70b-chat
- claim: This process continues until your account is closed or your balance is depleted.
- span (Evident Baseless Info): This process continues until your account is closed or your balance is depleted.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned* / Generative: "This process continues until your account is closed or your balance is depleted."

## 41. `ragtruth-15777-c8` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Holding a dressing or bandage in place.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 42. `ragtruth-15777-c9` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Hanging a picture or ornament.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 43. `ragtruth-15777-c10` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Repairing a ripped seam.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 44. `ragtruth-15777-c11` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Closing openings on a backpack or tent.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 45. `ragtruth-15777-c12` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Creating a makeshift fishing hook.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 46. `ragtruth-15777-c13` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Making a simple tool for cutting or prying.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 47. `ragtruth-15777-c14` — UNSOURCED, overlap 1.0
- model: llama-2-7b-chat
- claim: Creating a makeshift mirror or reflective surface.
- span (Evident Baseless Info): 6. Holding a dressing or bandage in place.
7. Hanging a picture or ornament.
8. Repairing a ripped seam.
9. Closing openings on a backpack or tent.
10. Creating a makeshift fishing hook.
11. Making a simple tool for cutting or prying.
12. Creating a makeshift mirror or reflective surface.
  - annotator note: HIGH INTRO OF NEW INFO / - Passages did not mention the additional 6 uses of safety pins (from 6 to 12)

## 48. `ragtruth-16056-c3` — UNSOURCED, overlap 1.0
- model: gpt-4-0613
- claim: Video guides showing what each skill move looks like are available at the top of the relevant page.
- span (Evident Baseless Info): Video guides showing what each skill move looks like are available at the top of the relevant page.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned in the source data* / Generative: "Video guides showing what each skill move looks like are available at the top of the relevant page."

## 49. `ragtruth-16510-c9` — UNSOURCED, overlap 1.0
- model: llama-2-13b-chat
- claim: Therefore, to get a toned body, you may need to prioritize building muscle through strength training and reducing body fat through a balanced diet and cardio exercises.
- span (Evident Baseless Info): Therefore, to get a toned body, you may need to prioritize building muscle through strength training and reducing body fat through a balanced diet and cardio exercises.
  - annotator note: HIGH INTRO OF NEW INFO / No mention of this information.

## 50. `ragtruth-16853-c4` — UNSOURCED, overlap 1.0
- model: llama-2-70b-chat
- claim: Structure: Red blood cells have a unique biconcave disk shape that allows them to maximize their surface area for gas exchange
- span (Evident Baseless Info): * Structure: Red blood cells have a unique biconcave disk shape that allows them to maximize their surface area for gas exchange, whereas white blood cells have a irregular, amoeba-like shape.
  - annotator note: HIGH INTRO OF NEW INFO / It includes information about the structure of red and white blood cells, which is not mentioned in the original source.

## 51. `ragtruth-16853-c5` — UNSOURCED, overlap 1.0
- model: llama-2-70b-chat
- claim: white blood cells have a irregular, amoeba-like shape.
- span (Evident Baseless Info): * Structure: Red blood cells have a unique biconcave disk shape that allows them to maximize their surface area for gas exchange, whereas white blood cells have a irregular, amoeba-like shape.
  - annotator note: HIGH INTRO OF NEW INFO / It includes information about the structure of red and white blood cells, which is not mentioned in the original source.

## 52. `ragtruth-16895-c3` — UNSOURCED, overlap 1.0
- model: llama-2-70b-chat
- claim: a darker yellow or amber color can indicate dehydration.
- span (Evident Baseless Info): a darker yellow or amber color can indicate dehydration.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned in the sourced data* / Generative: " a darker yellow or amber color can indicate dehydration. "

## 53. `ragtruth-16895-c7` — UNSOURCED, overlap 1.0
- model: llama-2-70b-chat
- claim: However, any persistent or significant changes in urine color should be evaluated by a healthcare professional to rule out potential health issues.
- span (Evident Baseless Info): However, any persistent or significant changes in urine color should be evaluated by a healthcare professional to rule out potential health issues.
  - annotator note: HIGH INTRODUCTION OF NEW INFORMATION / Original: *not mentioned in the sourced data* / Generative: " However, any persistent or significant changes in urine color should be evaluated by a healthcare professional to rule out potential health issues. "
