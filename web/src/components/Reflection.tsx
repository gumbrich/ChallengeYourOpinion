import { Check, Compass, ArrowRight } from 'lucide-react';
import { z } from 'zod';
import { useState } from 'react';
import { useLocalState } from '../hooks';
import type { Story } from '../data/schema';
const reflectionSchema = z.object({
  position: z.string(),
  note: z.string(),
  after: z.string(),
  completed: z.boolean(),
});
const empty = { position: '', note: '', after: '', completed: false };
export default function Reflection({ story }: { story: Story }) {
  const [reflection, setReflection, failed] = useLocalState(
    `weitblick:reflection:${story.id}`,
    empty,
    reflectionSchema,
  );
  const [step, setStep] = useState(reflection.completed ? 3 : 0);
  const choices = ['Eher ja', 'Noch unentschieden', 'Eher nein'];
  return (
    <section className="reflection" id="reflexion" aria-labelledby="reflection-title">
      <div className="reflection-intro">
        <div className="eyebrow">
          <Compass size={18} /> DEIN PERSPEKTIVWECHSEL
        </div>
        <h2 id="reflection-title">
          Nicht gewinnen. <br />
          <em>Weiterdenken.</em>
        </h2>
        <p>Deine Antworten bleiben in diesem Browser. Kein Profil, kein Meinungsscore.</p>
        <p className="small">
          {failed
            ? 'Speichern ist in diesem Browser nicht möglich. Deine Antworten bleiben nur für diese Sitzung.'
            : 'Du kannst deine Gedanken jederzeit löschen.'}
        </p>
      </div>
      <div className="reflection-content">
        <div className="step-progress" aria-label={`Schritt ${Math.min(step + 1, 3)} von 3`}>
          {['Standpunkt', 'Gegenargument', 'Zweiter Blick'].map((s, i) => (
            <span className={step >= i ? 'current' : ''} key={s}>
              {i + 1} {s}
            </span>
          ))}
        </div>
        <div aria-live="polite">
          {step === 0 ? (
            <>
              <h3>{story.question}</h3>
              <fieldset className="choices">
                <legend className="sr-only">Dein erster Standpunkt</legend>
                {choices.map((c) => (
                  <label className={reflection.position === c ? 'selected' : ''} key={c}>
                    <input
                      type="radio"
                      name="position"
                      value={c}
                      checked={reflection.position === c}
                      onChange={() => setReflection({ ...reflection, position: c })}
                    />
                    {c}
                  </label>
                ))}
              </fieldset>
              <button className="button" disabled={!reflection.position} onClick={() => setStep(1)}>
                Weiter zum Gegenargument <ArrowRight size={16} />
              </button>
            </>
          ) : step === 1 ? (
            <>
              <h3>Welches Gegenargument verdient einen zweiten Blick?</h3>
              <p>
                Wähle bewusst einen Gedanken, der deinen Standpunkt herausfordert. Was müsste
                stimmen, damit er dich überzeugt?
              </p>
              <label className="sr-only" htmlFor="reflection-note">
                Dein Gegenargument und deine Gedanken
              </label>
              <textarea
                id="reflection-note"
                maxLength={2000}
                rows={4}
                placeholder="Ein starkes Gegenargument ist …"
                value={reflection.note}
                onChange={(e) => setReflection({ ...reflection, note: e.target.value })}
              />
              <div className="action-row">
                <button className="text-link" onClick={() => setStep(0)}>
                  Zurück
                </button>
                <button
                  className="button"
                  disabled={!reflection.note.trim()}
                  onClick={() => setStep(2)}
                >
                  Noch einmal hinsehen <ArrowRight size={16} />
                </button>
              </div>
            </>
          ) : step === 2 ? (
            <>
              <h3>Hat sich dein Blick verändert?</h3>
              <fieldset className="choices vertical">
                <legend className="sr-only">Dein zweiter Blick</legend>
                {[
                  'Ich sehe meine Position differenzierter.',
                  'Ich bleibe dabei, verstehe die andere Seite besser.',
                  'Ich brauche weitere Informationen.',
                ].map((c) => (
                  <label className={reflection.after === c ? 'selected' : ''} key={c}>
                    <input
                      type="radio"
                      name="after"
                      checked={reflection.after === c}
                      onChange={() => setReflection({ ...reflection, after: c })}
                    />
                    {c}
                  </label>
                ))}
              </fieldset>
              <div className="action-row">
                <button className="text-link" onClick={() => setStep(1)}>
                  Zurück
                </button>
                <button
                  className="button"
                  disabled={!reflection.after}
                  onClick={() => {
                    setReflection({ ...reflection, completed: true });
                    setStep(3);
                  }}
                >
                  Reflexion abschließen <Check size={16} />
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="success-label">
                <Check size={18} /> Ein Gedanke weiter.
              </div>
              <h3>{reflection.after}</h3>
              <p>Dein Ausgangspunkt: {reflection.position}</p>
              <blockquote>{reflection.note}</blockquote>
              <button className="text-link" onClick={() => setStep(0)}>
                Gedanken bearbeiten <ArrowRight size={16} />
              </button>
            </>
          )}
        </div>
        {(reflection.position || reflection.note) && (
          <button
            className="delete-reflection"
            onClick={() => {
              setReflection(empty);
              setStep(0);
            }}
          >
            Meine Antworten löschen
          </button>
        )}
      </div>
    </section>
  );
}
